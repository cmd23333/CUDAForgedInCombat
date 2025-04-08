/*
    1. 每个 block 处理一个 (batch, head): gridDim.x = batch_size, gridDim.y = num_heads
    2. 每个线程处理 single_O 的多行: blockDim.x = O_tile_size, blockDim.y = 1, 用 for 循环处理 Depth 的每个维度
*/
#pragma once
#include <numeric>

namespace combat {

namespace flash_attention {

template<class T, int Bc, int Br>
__global__ kernel_flash_attention(
    T *device_out, T const *device_Q, T const *device_K, T const *device_V,
    T *device_sum, T *device_max,
    std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t depth
) {
    static_assert(Br==Bc, "暂时只实现了 Br=Bc 的情况");
    // blockDim.x == Br == Bc, 
    // 每个线程处理 O 的第 threadIdx.x + integral * Br 行, integral = 0, 1, 2, ..., (seq_len / Br) - 1
    int batch_index = blockIdx.x;
    int head_index = blockIdx.y;

    auto const offset_to_current_2d_matrix = batch_index * (num_heads * seq_len * depth) + head_index * (seq_len * depth);
    T const *single_Q = device_Q + offset_to_current_2d_matrix; // [seq_len, depth]
    T const *single_K = device_K + offset_to_current_2d_matrix;
    T const *single_V = device_V + offset_to_current_2d_matrix;
    T *single_O = device_out + offset_to_current_2d_matrix;
    
    T *single_sum = device_sum + offset_to_current_2d_matrix/depth; // [seq_len, 1]
    T *single_max = device_max + offset_to_current_2d_matrix/depth;

    // 动态共享内存, 内存大小为, 可以放下: 单个 q,k,v tile + attention_logit tile
    // 其中 q_tile 的 shape 为 [Br, depth]
    // k_tile, v_tile 的 shape 为 [Bc, depth]
    // attention_logit tile 的 shape 为 [Br, Bc]
    extern __shared__ T shm[];
    T *q_tile = shm;
    T *k_tile = shm + Br * depth;
    T *v_tile = shm + Br * depth + Bc * depth;
    T *att_tile = shm + Br * depth + 2 * Bc * depth;

    const int num_k_v_tiles = (seq_len + Bc - 1) / Bc; // k,v tile 的数量, 向上取整
    const int num_q_tiles = (seq_len + Br - 1) / Bc; // q tile 的数量, 向上取整

    for (int i=0; i<num_q_tiles; ++i) {
        // 把每行的初始 sum 赋值为 0, max 赋值为 T 类型的最小值
        device_sum[i*Br+threadIdx.x] = 0;
        device_max[i*Br+threadIdx.x] = std::numeric_limits<T>::lowest();
    }

    // 外层对 K 和 V 遍历
    for (int k_v_tile_index=0; k_v_tile_index<num_k_v_tiles; ++k_v_tile_index) {
        for (int di=0; di<depth; ++di) {
            // load k, v tile: [Bc, depth]. 我们要求 blockDim.x == Bc, 因此每个线程负责 tile 的一行刚好
            // 每个线程 load 一行 (一个 token 的所有 depth)
            k_tile[threadIdx.x * depth + di] = single_K[k_v_tile_index * Bc * depth + threadIdx.x * depth + di];
            v_tile[threadIdx.x * depth + di] = single_V[k_v_tile_index * Bc * depth + threadIdx.x * depth + di];
        }
        __syncthreads();

        // 内层对 Q 循环
        for (int q_tile_index=0; q_tile_index<num_q_tiles; ++q_tile_index) {
            for (int di=0; di<depth; ++di) {
                // load q tile: [Br, depth], 我们要求 blockDim.x == Br, 因此每个线程负责 tile 的一行刚好
                // 每个线程 load 一行 (一个 token 的所有 depth)
                q_tile[threadIdx.x * depth + di] = single_Q[q_tile_index * Br * depth + threadIdx.x * depth + di];
            }
            // __syncthreads(); 这里可以不用同步, 因为 q_tile 的一行足够算 a_tile/o_tile 的一行了(k,v_tile 已经提前 load 好)

            // 计算 a tile: [Br, Bc], 我们要求 blockDim.x == Br, 因此每个线程负责 tile 的一行刚好
            // 注意 a_tile = q_tile @ k_tile.T, 但我们并不会真正对 k_tile 做转置
            for (int j=0; j<Bc; ++j) {
                T a_tile_ij = 0;    // i -> threadIdx.x
                for (int di=0; di<depth; ++di) {
                    a_tile_ij += q_tile[threadIdx.x * depth + di] * k_tile[j * depth + di];
                }
                a_tile[threadIdx.x * Bc + j] = a_tile_ij / std::sqrt(depth);
            }
            // __syncthreads(); 这里也不用同步, 原因同上

            // 计算新的 max(随着 k/a_tile 往右移动), 其实每个 a_tile_ij 算完就可以做. 这里把他弄出来了
            T new_max = single_max[q_tile_index * Br + threadIdx.x];
            for (int j=0; j<Bc; ++j) {
                // 如果知道类型, 可以用内置函数加速, 比如固定 __fmaxf() 或者用 if constexpr (std::is_same_v<>) 写分支
                new_max = std::max(a_tile[threadIdx.x * Bc + j], new_max);   
            }
            
            // 计算新的 sum(随着 k/a_tile 往右移动)
            T new_sum = single_sum[q_tile_index * Br + threadIdx.x];
            // 先根据 max 调整原有的 sum
            T sum_adjust_coef = std::exp(single_max[q_tile_index * Br + threadIdx.x] - new_max);
            new_sum *= sum_adjust_coef;
            for (int j=0; j<Bc; ++j) {
                new_sum += std::exp(a_tile[threadIdx.x * Bc + j] - new_sum);
            }
            
            // 将结果更新到 O_tile(直接写入 O)
            // 根据矩阵乘法的数学原理, 需要更新 num_k_v_tiles 次
            T *o_tile = single_O[q_tile_index * Br * depth]
            for (int di=0; di<depth; ++di) {
                // 数学: o[i][j] = softmax(a[i][k]) * v[k][j], for k in range(Bc)
                // 代码: i=threadIdx.x, j=di
                T o_ij = 0;
                for (int k=0; k<Bc; ++k)
                    o_ij += std::exp(a_tile[threadIdx.x * Bc + k] - new_max) / new_sum * v_tile[k * depth + di]
                
                o_tile[threadIdx.x * depth + di] += o_ij;
            }

            single_max[q_tile_index * Br + threadIdx.x] = new_max;  // 外层循环变化时(k_tile 往右走时), 还会用到这行的 max
            single_sum[q_tile_index * Br + threadIdx.x] = new_sum;
        }

        __syncthreads();
    }
}

template<class T>
void flash_attention_cuda(
    T *out, T const *Q, T const *K, T const *V, 
    std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t depth
) {
    /*
        Q,K,V.shape = [batch, num_head, seq_len, depth]
    */
    auto const num_element = batch_size * num_heads * seq_len * depth;
    auto const sizeof_tensor = num_element * sizeof(T);

    T *device_out;
    T *device_Q, *device_K, *device_V, *device_sum, *device_max;
    cudaMalloc(&device_out, sizeof_tensor);
    cudaMalloc(&device_Q, sizeof_tensor); cudaMemcpy(device_Q, Q, sizeof_tensor, cudaMemcpyDefault);
    cudaMalloc(&device_K, sizeof_tensor); cudaMemcpy(device_K, K, sizeof_tensor, cudaMemcpyDefault);
    cudaMalloc(&device_V, sizeof_tensor); cudaMemcpy(device_V, V, sizeof_tensor, cudaMemcpyDefault);
    cudaMalloc(&device_max, sizeof_tensor/depth);
    cudaMalloc(&device_sum, sizeof_tensor/depth);

    constexpr int TileSize = 32;
    dim3 threads = {TileSize, 1, 1};
    dim3 blocks = {batch_size, num_heads, 1};

    kernel_flash_attention<T, TileSize, TileSize><<<blocks, threads>>>(
        device_out, device_Q, device_K, device_V,
        device_sum, device_max,
        batch_size, num_heads, seq_len, depth
    );

    cudaMemcpy(out, device_out, sizeof_tensor, cudaMemcpyDefault);
    cudaFree(device_out);
    cudaFree(device_Q);
    cudaFree(device_K);
    cudaFree(device_V);
    cudaFree(device_max);
    cudaFree(device_sum);
}

} // namespace flash_attention

} // namespace combat