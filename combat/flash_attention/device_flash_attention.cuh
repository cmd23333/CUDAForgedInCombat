#pragma once

#include <string.h>
#include <cmath>
#include <numeric>

#include "tools/show.hpp"
#include "combat/matrix_multiply/host_matrix_multiply.hpp"

namespace combat {

namespace flash_attention {

template<class T>
__device__ void scaled_matmul(T *out, T const *a, T const *b, std::size_t m, std::size_t k, std::size_t n) {
    /*
        a: [m, k], b: [n, k]
        calculate a*b.T: [m, n]
    */
    auto out = std::vector<T>(m*n, 0);

    for (std::size_t i=0; i<m; ++i) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t k_=0; k_<k; ++k_) {
                out[i*n+j] += a[i*k+k_] * b[j*k+k_];
            }
            out[i*n+j] /= std::sqrt(k);
        }
    }

    return out;
}

template<class T>
__device__ void online_softmax(
    int shift_to_q_or_o_tile,
    T *o_tile, T *a_tile, T const *q_tile, T const *k_tile, T const *v_tile,
    T *old_max, T *old_sum,
    std::size_t q_tile_height, std::size_t depth, std::size_t k_tile_height
) {
    memcpy(new_max, old_max, q_tile_height*sizeof(T));

    const auto a_tile_height = q_tile_height;
    const auto a_tile_width = k_tile_height;

    for (std::size_t a_tile_row_index=0; a_tile_row_index < a_tile_height; ++a_tile_row_index) {
        for (std::size_t a_tile_col_index=0; a_tile_col_index < a_tile_width; ++a_tile_col_index) {
            new_max[a_tile_row_index] = std::max(new_max[a_tile_row_index], a_tile[a_tile_row_index * a_tile_width + a_tile_col_index]);
        }
    }

    for (std::size_t a_tile_row_index=0; a_tile_row_index < a_tile_height; ++a_tile_row_index) {
        new_sum[a_tile_row_index] = 0;
        T const sum_adjust_coef = std::exp(old_max[a_tile_row_index] - new_max[a_tile_row_index]);
        new_sum[a_tile_row_index] = sum_adjust_coef * old_sum[a_tile_row_index];
        for (std::size_t a_tile_col_index=0; a_tile_col_index < a_tile_width; ++a_tile_col_index) {
            new_sum[a_tile_row_index] += std::exp(a_tile[a_tile_row_index * a_tile_width + a_tile_col_index] - new_max[a_tile_row_index]);
        }
    }

    tools::show_matrix(new_max, 1, q_tile_height, "max");
    tools::show_matrix(new_sum, 1, q_tile_height, "sum");

    const auto o_tile_height = q_tile_height;
    const auto o_tile_width = depth;
    const auto v_tile_height = k_tile_height;
    const auto v_tile_width = depth;
    for (std::size_t o_tile_row_index=0; o_tile_row_index < o_tile_height; ++o_tile_row_index) {
        // local sum: [q_tile_height, 1]
        auto const o_adjust_coef = std::exp(old_max[o_tile_row_index] - new_max[o_tile_row_index]) * old_sum[o_tile_row_index] / new_sum[o_tile_row_index];
        for (std::size_t o_tile_col_index=0; o_tile_col_index < o_tile_width; ++o_tile_col_index) {
            o_tile[o_tile_row_index * depth + o_tile_col_index] = o_tile[o_tile_row_index * depth + o_tile_col_index] * o_adjust_coef;
            
            T o_ij = 0;
            // a_tile_width == k_tile_height == v_tile_height
            // a_tile: [m, n], v_tile: [n, d]
            // o_tile: [m, d]
            for (std::size_t a_tile_col_index=0; a_tile_col_index < a_tile_width; ++a_tile_col_index) {
                T const a_tile_ik = std::exp(a_tile[o_tile_row_index * a_tile_width + a_tile_col_index] - new_max[o_tile_row_index]) / new_sum[o_tile_row_index];
                o_ij += a_tile_ik * v_tile[a_tile_col_index * v_tile_width + o_tile_col_index];
            }

            o_tile[o_tile_row_index * depth + o_tile_col_index] += o_ij;
        }
    }

    memcpy(old_max, new_max, q_tile_height*sizeof(T));
    memcpy(old_sum, new_sum, q_tile_height*sizeof(T));
    free(new_max);
    free(new_sum);
}


template<class T>
__global__ void kernel_online_single_head_attention(
    T *O, T const *Q, T const *K, T const *V, 
    std::size_t seq_len, std::size_t depth
) {
    /*
        gridDim.x = 1
        gridDim.y = seq_len / tile_size
        blockDim.x == depth
        blockDim.y == tile_size
    */
    constexpr int tile_size = 4;
    auto const tile_width = depth;

    __shared__ T q_tile[tile_size][depth];
    __shared__ T k_tile[tile_size][depth];
    __shared__ T v_tile[tile_size][depth];
    __shared__ T o_tile[tile_size][depth];
    __shared__ T a_tile[tile_size][tile_size];
    __shared__ T sum_[seq_len];
    __shared__ T max_[seq_len];
    __shared__ T tile_sum_[tile_size];
    __shared__ T tile_max_[tile_size];

    // q, k, v, o 的 tiles 个数一致
    size_t const num_tiles = seq_len / tile_size;

    auto const shift_to_o_tile = blockIdx.y * tile_size * depth;
    auto const tile_x = threadIdx.x; // [0, depth-1]
    auto const tile_y = threadIdx.y; // [0, tile_size-1]

    if (tile_x == 0) {
        max_[tile_y] = std::numeric_limits<T>::lowest();
        sum_[tile_y] = 0;
    }

    // 我们保证 num_k_tile == num_v_tile == num_q_tile == num_o_tile
    for (size_t k_or_v_tile_index=0; k_or_v_tile_index<num_tiles; ++k_or_v_tile_index) {
        // 外层循环, 取出 k_tile 和 v_tile; tile_width == depth
        auto const shift_to_k_or_v_tile = k_or_v_tile_index * v_tile_height * tile_width;
        k_tile[tile_y][tile_x] = single_K[shift_to_k_or_v_tile + tile_y * tile_width + tile_x];
        v_tile[tile_y][tile_x] = single_V[shift_to_k_or_v_tile + tile_y * tile_width + tile_x];
        __syncthreads();

        for (size_t q_or_v_tile_index=0; q_or_v_tile_index < num_tiles; ++q_or_v_tile_index) {
            // 内层循环, 取出 q_tile 和 o_tile
            auto const shift_to_q_or_o_tile = q_or_v_tile_index * tile_size * tile_width;
            q_tile[tile_y][tile_x] = single_Q[shift_to_q_or_o_tile + tile_y * tile_width + tile_x];
            __syncthreads();
            
            a_tile[tile_y][tile_x] = 0;
            for (int k=0; k<depth; ++k)
                // 注意我们没有对 k_tile 做转置
                a_tile[tile_y][tile_x] += q_tile[tile_y][k] * k_tile[tile_x][k];
            a_tile[tile_y][tile_x] /= std::sqrt(depth);
            __syncthreads();
            
            if (tile_x == 0) 
                tile_max_[tile_y] = old_max[q_or_v_tile_index * tile_size + tile_y];

            const auto a_tile_height = q_tile_height;
            const auto a_tile_width = k_tile_height;
        
            for (std::size_t a_tile_row_index=0; a_tile_row_index < a_tile_height; ++a_tile_row_index) {
                for (std::size_t a_tile_col_index=0; a_tile_col_index < a_tile_width; ++a_tile_col_index) {
                    new_max[a_tile_row_index] = std::max(new_max[a_tile_row_index], a_tile[a_tile_row_index * a_tile_width + a_tile_col_index]);
                }
            }
        }
    }
}

template<class T>
void flash_attention_cuda(
    T *out, T const *Q, T const *K, T const *V, 
    std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t depth
) {
    /*
        提示: Q,K,V.shape = [batch, num_head, seq_len, depth]
    */
    auto const num_element = batch_size * num_heads * seq_len * depth;
    auto const sizeof_tensor = num_element * sizeof(T);

    T *device_out, const *device_Q, const *device_K, const *device_V;
    cudaMalloc(&device_out, sizeof_tensor);
    cudaMalloc(&device_Q, sizeof_tensor); cudaMemcpy(device_Q, Q, sizeof_tensor, cudaMemcpyDefault);
    cudaMalloc(&device_K, sizeof_tensor); cudaMemcpy(device_K, K, sizeof_tensor, cudaMemcpyDefault);
    cudaMalloc(&device_V, sizeof_tensor); cudaMemcpy(device_V, V, sizeof_tensor, cudaMemcpyDefault);

    for (size_t batch_index=0; batch_index < batch_size; ++batch_index) {
        for (size_t head_index=0; head_index < num_heads; ++head_index) {
            // 一开始我觉得可以用 blocks = {num_heads, batch_size, 1} 每个 block 处理一个 (batch, head)
            // 想多了... 还是得循环
            auto const shift_to_current_2d_matrix = batch_index * (num_heads * seq_len * depth) + head_index * (seq_len * depth);
            T const *single_Q = Q + shift_to_current_2d_matrix; // [seq_len, depth]
            T const *single_K = K + shift_to_current_2d_matrix;
            T const *single_V = V + shift_to_current_2d_matrix;
            T *single_O = out + shift_to_current_2d_matrix;

            constexpr int tile_size = 4;
            // 每个 block 处理一个结果的 o_tile
            dim3 threads = {depth, tile_size, 1};
            // 感觉很奇怪啊 ... 因为 threads 的个数是有限制的, 如果 depth * tile_size 太大了怎么办?
            dim3 blocks = {1 , (seq_len + threads.y - 1) / threads.y, 1};

            kernel_online_single_head_attention<<<blocks, threads>>>(single_O, single_Q, single_K, single_V, seq_len, depth);
        }
    }
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    std::vector<T> out(num_element, 0);
    cudaMemcpy(out.data(), device_out, sizeof_tensor, cudaMemcpyDefault);
    cudaFree(device_out);
    cudaFree(device_Q);
    cudaFree(device_K);
    cudaFree(device_V);

    return out;
}

} // namespace flash_attention

} // namespace combat