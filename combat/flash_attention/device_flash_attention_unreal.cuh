/*
    计算结果是对的, 也用到了 CUDA, 但这个实现方式是错误的
    每次 launch kernel 只计算一个 batch, head
    并且每个 block 只管一个 O_tile 的计算结果, 有大量资源浪费
*/

#pragma once

#include <string.h>
#include <cmath>
#include <numeric>

#include "tools/show.hpp"
#include "combat/matrix_multiply/host_matrix_multiply.hpp"

namespace combat {

namespace flash_attention {

namespace unreal {
// CUDA 不支持 float 的 atomicMax
__device__ float atomicMax(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}


template<class T, int TileSize=4, int Depth=16>
__global__ void kernel_online_single_head_attention(
    T *single_O, T const *single_Q, T const *single_K, T const *single_V, 
    std::size_t seq_len
) {
    /*
        gridDim.x = 1
        gridDim.y = seq_len / tile_size
        blockDim.x == depth
        blockDim.y == tile_size

        每个 block 计算一个 o_tile
    */

    __shared__ T q_tile[TileSize][Depth];
    __shared__ T k_tile[TileSize][Depth];
    __shared__ T v_tile[TileSize][Depth];
    __shared__ T a_tile[TileSize][TileSize];

    __shared__ T sum_tile[TileSize]; // 存这个 Tile 的 max/sum
    __shared__ T max_tile[TileSize];

    // q, k, v, o 的 tiles 个数一致
    size_t const num_tiles = seq_len / TileSize;

    auto const tile_x = threadIdx.x; // [0, depth-1]
    int const tile_y = threadIdx.y; // [0, tile_size-1]

    if (tile_x == 0) {
        max_tile[tile_y] = std::numeric_limits<T>::lowest();    // 需要开启 --expt-relaxed-constexpr
        sum_tile[tile_y] = 0;
    }

    // 为什么写到这里我觉得不对了呢, 因为每个 block 处理一个 O_tile 那他只需要读取对应那行的一个 Q_tile, 和论文说的不对
    int q_or_o_tile_index = blockIdx.y;
    auto const shift_to_q_or_o_tile = q_or_o_tile_index * TileSize * Depth;
    q_tile[tile_y][tile_x] = single_Q[shift_to_q_or_o_tile + tile_y * Depth + tile_x];
    T *o_tile = single_O + shift_to_q_or_o_tile;
    o_tile[tile_y * Depth + tile_x] = 0;
    __syncthreads();

    // 我们保证 num_k_tile == num_v_tile == num_q_tile == num_o_tile
    for (std::size_t k_or_v_tile_index=0; k_or_v_tile_index<num_tiles; ++k_or_v_tile_index) {
        // 外层循环, 取出 k_tile 和 v_tile; tile_width == depth
        auto const shift_to_k_or_v_tile = k_or_v_tile_index * TileSize * Depth;
        auto const unsqueezed_index = shift_to_k_or_v_tile + tile_y * Depth + tile_x;
        k_tile[tile_y][tile_x] = single_K[unsqueezed_index];
        v_tile[tile_y][tile_x] = single_V[unsqueezed_index];
        __syncthreads();

        // 共 depth * tile_size 个线程, a_tile[TileSize][TileSize]
        // 假设 Depth > TileSize
        if (tile_x < TileSize) {
            a_tile[tile_y][tile_x] = 0;
            for (int k=0; k<Depth; ++k)
                // 注意我们没有对 k_tile 做转置
                a_tile[tile_y][tile_x] += q_tile[tile_y][k] * k_tile[tile_x][k];
            a_tile[tile_y][tile_x] /= std::sqrt(Depth);
        }
        if (false && threadIdx.x==0 && threadIdx.y==0 && k_or_v_tile_index == 0 && blockIdx.y == 0) {
            printf("%f, %f, %f, %f \n", a_tile[0][0], a_tile[0][1], a_tile[0][2], a_tile[0][3]);
            printf("%f, %f, %f, %f \n", a_tile[1][0], a_tile[1][1], a_tile[1][2], a_tile[1][3]);
            printf("%f, %f, %f, %f \n", a_tile[2][0], a_tile[2][1], a_tile[2][2], a_tile[2][3]);
            printf("%f, %f, %f, %f \n", a_tile[3][0], a_tile[3][1], a_tile[3][2], a_tile[3][3]);
        }
        __syncthreads();

        __shared__ T local_sum[TileSize];  // new_sum
        __shared__ T local_max[TileSize];

        if (tile_x == 0) {
            // 取出历史这几行的 old_max
            local_max[tile_y] = max_tile[tile_y];
            local_sum[tile_y] = 0;
        }
        __syncthreads();
        if (tile_x < TileSize) {
            int a_tile_row_index = tile_y;
            int a_tile_col_index = tile_x;
            atomicMax(&local_max[a_tile_row_index], a_tile[a_tile_row_index][a_tile_col_index]);
        }
        __syncthreads();

        if (tile_x == 0) {
            T sum_adjust_coef = std::exp(max_tile[tile_y] - local_max[tile_y]);
            local_sum[tile_y] = sum_adjust_coef * sum_tile[tile_y];
        }
        __syncthreads();

        if (tile_x < TileSize) {
            int a_tile_row_index = tile_y;
            int a_tile_col_index = tile_x;
            // 感觉很慢?
            atomicAdd(&local_sum[a_tile_row_index], std::exp(a_tile[a_tile_row_index][a_tile_col_index] - local_max[a_tile_row_index]));
        }
        __syncthreads();

        if (false && threadIdx.x==0 && threadIdx.y==0 && k_or_v_tile_index == 0 && blockIdx.y == 0) {
            printf("sum %f, %f, %f, %f \n", local_sum[0], local_sum[1], local_sum[2], local_sum[3]);
            printf("max %f, %f, %f, %f \n", local_max[0], local_max[1], local_max[2], local_max[3]);
        }
    
        int o_tile_row_index = tile_y;
        int o_tile_col_index = tile_x;
        // local sum: [q_tile_height, 1]
        auto const o_adjust_coef = std::exp(max_tile[o_tile_row_index] - local_max[o_tile_row_index]) * sum_tile[o_tile_row_index] / local_sum[o_tile_row_index];
        o_tile[o_tile_row_index * Depth + o_tile_col_index] = o_tile[o_tile_row_index * Depth + o_tile_col_index] * o_adjust_coef;

        T o_ij = 0;
        for (std::size_t a_tile_col_index=0; a_tile_col_index < TileSize; ++a_tile_col_index) {
            T const a_tile_ik = std::exp(a_tile[o_tile_row_index][a_tile_col_index] - local_max[o_tile_row_index]) / local_sum[o_tile_row_index];
            o_ij += a_tile_ik * v_tile[a_tile_col_index][o_tile_col_index];
        }
        
        o_tile[o_tile_row_index * Depth + o_tile_col_index] += o_ij;
        __syncthreads();

        if (tile_x == 0) {
            max_tile[tile_y] = local_max[tile_y];
            sum_tile[tile_y] = local_sum[tile_y];
        }
        __syncthreads();
    }
}

template<class T, int Depth>
void flash_attention_cuda(
    T *out, T const *Q, T const *K, T const *V, 
    std::size_t batch_size, std::size_t num_heads, std::size_t seq_len
) {
    /*
        提示: Q,K,V.shape = [batch, num_head, seq_len, depth]
    */
    auto const num_element = batch_size * num_heads * seq_len * Depth;
    auto const sizeof_tensor = num_element * sizeof(T);

    T *device_out;
    T *device_Q, *device_K, *device_V;
    cudaMalloc(&device_out, sizeof_tensor);
    cudaMalloc(&device_Q, sizeof_tensor); cudaMemcpy(device_Q, Q, sizeof_tensor, cudaMemcpyDefault);
    cudaMalloc(&device_K, sizeof_tensor); cudaMemcpy(device_K, K, sizeof_tensor, cudaMemcpyDefault);
    cudaMalloc(&device_V, sizeof_tensor); cudaMemcpy(device_V, V, sizeof_tensor, cudaMemcpyDefault);

    for (size_t batch_index=0; batch_index < batch_size; ++batch_index) {
        for (size_t head_index=0; head_index < num_heads; ++head_index) {
            // 一开始我觉得可以用 blocks = {num_heads, batch_size, 1} 每个 block 处理一个 (batch, head)
            // 想多了... 还是得循环
            // 不过应该可以开多个 stream 来并行
            auto const shift_to_current_2d_matrix = batch_index * (num_heads * seq_len * Depth) + head_index * (seq_len * Depth);
            T const *single_Q = device_Q + shift_to_current_2d_matrix; // [seq_len, depth]
            T const *single_K = device_K + shift_to_current_2d_matrix;
            T const *single_V = device_V + shift_to_current_2d_matrix;
            T *single_O = device_out + shift_to_current_2d_matrix;

            constexpr int TileSize = 4;
            // 每个 block 处理一个结果的 o_tile
            dim3 threads = {Depth, TileSize, 1};
            // 感觉很奇怪啊 ... 因为 threads 的个数是有限制的, 如果 depth * tile_size 太大了怎么办?
            // emm 应该不会? 考虑到多头, 每个头的 depth 会比较小
            dim3 blocks = {1 , (seq_len + threads.y - 1) / threads.y, 1};

            std::cout << "threads.xyz = " << threads.x << ", " << threads.y << ", " << threads.z << std::endl;
            std::cout << "blocks.xyz = " << blocks.x << ", " << blocks.y << ", " << blocks.z << std::endl;

            kernel_online_single_head_attention<T, TileSize, Depth><<<blocks, threads>>>(single_O, single_Q, single_K, single_V, seq_len);
            cudaDeviceSynchronize();
            std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
        }
    }

    cudaMemcpy(out, device_out, sizeof_tensor, cudaMemcpyDefault);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaFree(device_out);
    cudaFree(device_Q);
    cudaFree(device_K);
    cudaFree(device_V);
}

} // namespace unreal

} // namespace flash_attention

} // namespace combat