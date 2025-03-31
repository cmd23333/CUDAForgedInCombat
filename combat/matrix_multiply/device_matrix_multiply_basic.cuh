#pragma once

#include <vector>
#include <iostream>
#include <chrono>
#include <numeric>

#include "cuda_runtime.h"
#include "thrust.h"
#include "tools/math.hpp"

namespace combat {

namespace matrix_multiply {
/*
    2.2 cuda 矩阵乘（提示：复用 host 的逻辑，解释为什么缓存更高效）
    2.3 cuda 矩阵乘（提示：分块+共享内存）
*/

template <class T>
__global__ void kernel_matrix_multiply_trival(
    T *out, T const* mat1, T const *mat2,
    std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width
) {
    /*
        复用 host 的逻辑
        为什么缓存更高效? 因为大量的线程在并行, Cache 命中率高
    */
    std::size_t out_i = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t out_j = blockIdx.x * blockDim.x + threadIdx.x;

    for (std::size_t k=0; k<mat1_width; ++k) 
        out[out_i * mat2_width + j] += mat1[out_i * mat1_width + k] * mat2[k * mat2_width + out_j];
}

template <class T>
std::vector<T> matrix_multiply_trival(
    T const* mat1, T const *mat2, 
    std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width
) {
    auto device_out = thrust::device_vector<T>(0, mat1_height * mat2_width);

    dim3 threads = {32, 32, 1};
    dim3 blocks = {(mat2_width + threads.x - 1) / threads.x, (mat1_height + threads.y - 1) / threads.y, 1};

    kernel_matrix_multiply_trival<<<blocks, threads>>>(device_out.data().get(), mat1, mat2, mat1_height, mat1_width, mat2_width);
    
    // 自动同步 + 复制数据到 host
    std::vector<T> out = device_out;
}

template <class T, int TileSize = 16>
__global__ void kernel_matrix_multiply_tiling(
    T *out, T const* mat1, T const *mat2,
    std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width
) {
    /*
        使用分块相乘, 用共享内存减少读取次数
        假设矩阵长宽都可以被 16 整除
    */

    __shared__ T shared_tile_in_mat1[TileSize][TileSize];
    __shared__ T shared_tile_in_mat2[TileSize][TileSize];

    int row_index_in_tile = threadIdx.y;
    int col_index_in_tile = threadIdx.x;

    // 此 block 计算出的 Tile 的左上角坐标
    int y_out_tile = blockIdx.y * blockDim.y;
    int x_out_tile = blockIdx.x * blockDim.x;

    int row_index_in_out = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index_in_out = blockIdx.x * blockDim.x + threadIdx.x;

    // 可以保证这两个是相等的
    int number_tiles_in_mat1_per_row = mat1_width / TileSize;
    int number_tiles_in_mat2_per_col = mat1_width / TileSize;

    // 最终存储在 out[row_index_in_out][col_index_in_out] 的计算结果
    T element_value_in_out;
    for (int tile_index=0; tile_index < number_tiles_in_mat1_per_row; ++tile_index) {
        // mat1_tile 沿着 mat1 的行移动
        T const *mat1_tile_upper_left = mat1 + y_out_tile * mat1_width + tile_index * TileSize;
        // mat2_tile 沿着 mat2 的列移动
        T const *mat2_tile_upper_left = mat2 + tile_index * TileSize * mat2_width + x_out_tile * TileSize;
        
        shared_tile_in_mat1[row_index_in_tile][col_index_in_tile] = mat1[mat1_tile_upper_left + row_index_in_tile * mat1_width + col_index_in_out];
        shared_tile_in_mat2[row_index_in_tile][col_index_in_tile] = mat2[mat2_tile_upper_left + row_index_in_tile * mat2_width + col_index_in_out];

        __syncthreads();

        for (int k=0; k<TileSize; ++k)
            element_value_in_out += shared_tile_in_mat1[row_index_in_tile][k] * shared_tile_in_mat2[k][col_index_in_tile];
    }
    out[row_index_in_out * mat2_width + col_index_in_out] = element_value_in_out;
}

template <class T>
std::vector<T> matrix_multiply_trival(
    T const* mat1, T const *mat2, 
    std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width
) {
    auto device_out = thrust::device_vector<T>(0, mat1_height * mat2_width);

    constexpr int TileX = 16;
    constexpr int tileY = 16;
    dim3 threads = {TileX, tileY, 1};
    dim3 blocks = {(mat2_width + threads.x - 1) / threads.x, (mat1_height + threads.y - 1) / threads.y, 1};

    kernel_matrix_multiply_tiling<T, TileX, tileY><<<blocks, threads>>>(
        device_out.data().get(), mat1, mat2, mat1_height, mat1_width, mat2_width
    );
    
    // 自动同步 + 复制数据到 host
    std::vector<T> out = device_out;
}

} // namespace matrix_multiply

} // namespace combat
