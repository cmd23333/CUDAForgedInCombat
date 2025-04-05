#pragma once

#include <vector>
#include <mma.h>

#include "cuda_fp16.h"

namespace combat {

namespace matrix_multiply {

template<class T1, class T2, typename _ =  std::enable_if<std::is_same_v<T2, half>>>
__global__ void kernel_matmul_tc(
    T1 *out, T2 const* mat1, T2 const *mat2,
    std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width
) {
    //假设out可以被16x16的Tile完全分块
    //每个Warp计算出一个out的Tile
    constexpr int tile_size = 16;
    constexpr int warp_size = 32;
    int const warp_index = (threadIdx.x + blockDim.x* blockIdx.x) / warp_size;

    if (warp_index * 16 * 16 >= mat1_height * mat2_width)
        // in case 矩阵太小
        return;

    int const num_tiles_per_row_in_out = mat2_width / tile_size;
    int const tile_row_index_in_out = warp_index / num_tiles_per_row_in_out;
    int const tile_col_index_in_out = warp_index % num_tiles_per_row_in_out;

    int const num_tiles_per_row_in_mat1 = mat1_width/ tile_size;
    int shift_to_mat1_first_row_tile = tile_row_index_in_out * tile_size * mat1_width;
    int shift_to_mat2_first_col_tile = tile_col_index_in_out * tile_size;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T1> frag_c;
    nvcuda::wmma::fill_fragment(frag_c, 0);

    for (int i=0; i<num_tiles_per_row_in_mat1; ++i) {
        int shift_to_mat1 = shift_to_mat1_first_row_tile+ i * tile_size;
        int shift_to_mat2 = shift_to_mat2_first_col_tile + i * tile_size * mat2_width;
        nvcuda::wmma::load_matrix_sync(frag_a, &mat1[shift_to_mat1], mat1_width);
        nvcuda::wmma::load_matrix_sync(frag_b, &mat2[shift_to_mat2], mat2_width);
        // C= a @ b + C, 如果 satf 设置为 true, 那么溢出被设置为 ±max_norm, NaN 被设置为 0
        // nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c, false);   奇怪的是我的 4070 + CUDA 12.8 并没有这个参数
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    // 注意这里最后的参数是 nvcuda::wmma::mem_row_major
    nvcuda::wmma::store_matrix_sync(
        &out[tile_row_index_in_out * 16 * mat2_width + tile_col_index_in_out], frag_c, mat2_width, nvcuda::wmma::mem_row_major
    );
}

template<class T1, class T2, typename _ = std::enable_if<std::is_same_v<T2, half>>>
__global__ void kernel_matmul_tc_shm(
    T1 *out, T2 const* mat1, T2 const *mat2,
    std::size_t mat1_height, std::size_t mat1_width,std::size_t mat2_width
) {
    /*
        保证调用时是 <<<..., 256>>>
        使用共享内存, 每个 block 可以共享 A 的 tile
        需要保证每个 block 都处理 out 里同一行的 tile
        
        反例, 比如 out: [64, 32], blockDim.x = 256 = 8个线程束
        前2个处理 out 的 [:16, :]
        如果我们往共享内存里保存了 sm[16][16] = out[:16,:16], 然后同步
        实际上这个 sm 只有前两个束计算的应该用到

        TODO: 应该改一下... 太小的就不要用 shared memory 了, corner case 太多
    */
    constexpr int warp_size = 32;
    int const warp_index = (threadIdx.x + blockDim.x* blockIdx.x) / warp_size;

    constexpr int tile_size = 16;
    __shared__ T2 shared_a_tile[tile_size][tile_size];

    int const num_tiles_per_row_in_out = mat2_width / tile_size;
    if (num_tiles_per_row_in_out != 1 && num_tiles_per_row_in_out % 8 != 0) {
        // 保证 blockDim.x == 256 的情况下, 如果每行不是 8 整数倍的 Tile 数, 说明不合规
        // 只有 1 个 tile 除外
        return;
    }

    int const tile_row_index_in_out = num_tiles_per_row_in_out > 8 ? warp_index / num_tiles_per_row_in_out : 0;
    int const tile_col_index_in_out = num_tiles_per_row_in_out > 8 ? warp_index % num_tiles_per_row_in_out : 0;

    int const num_tiles_per_row_in_mat1 = mat1_width / tile_size;
    int shift_to_mat1_first_row_tile = tile_row_index_in_out * tile_size * mat1_width;
    int shift_to_mat2_first_col_tile = tile_col_index_in_out * tile_size;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T1> frag_c;
    nvcuda::wmma::fill_fragment(frag_c, 0);

    if (threadIdx.x==0 && blockIdx.x==0)
        printf("num_tiles_per_row_in_mat1 %d \n", num_tiles_per_row_in_mat1);

    for (int i=0; i<num_tiles_per_row_in_mat1; ++i) {
        int shift_to_mat1 = shift_to_mat1_first_row_tile + i * tile_size;
        int sy = threadIdx.x/tile_size;
        int sx = threadIdx.x%tile_size;
        shared_a_tile[sy][sx] = mat1[shift_to_mat1+sy*mat1_width+sx];
        if (threadIdx.x==255 && blockIdx.x==0) {
            printf("mat1[%d][%d] = %f \n", sy, sx, __half2float(shared_a_tile[sy][sx]));
            printf("shift_to_mat1 %d, mat1[15][15] = %f \n", shift_to_mat1, __half2float(mat1[shift_to_mat1 + 15*16+15]));
        }
        __syncthreads();

        if (warp_index * 16 * 16 >= mat1_height * mat2_width)
            // in case 矩阵太小, 但是给共享 tile 做贡献还是要做的
            return;

        int shift_to_mat2 = shift_to_mat2_first_col_tile + i * tile_size * mat2_width;
        if (threadIdx.x==0 && blockIdx.x==0) {
            printf("shift_to_mat1 %d \n", shift_to_mat1);
            printf("shift_to_mat2 %d \n", shift_to_mat2);
            printf("shared_a_tile[0][0] %f \n", __half2float(shared_a_tile[0][0]));
            printf("shared_a_tile[1][0] %f \n", __half2float(shared_a_tile[1][0]));
            printf("shared_a_tile[2][0] %f \n", __half2float(shared_a_tile[2][0]));
            printf("shared_a_tile[3][0] %f \n", __half2float(shared_a_tile[3][0]));
            printf("shared_a_tile[4][0] %f \n", __half2float(shared_a_tile[4][0]));
        }

        nvcuda::wmma::load_matrix_sync(frag_a, &shared_a_tile[0][0], tile_size);
        nvcuda::wmma::load_matrix_sync(frag_b, &mat2[shift_to_mat2], mat2_width);
        // C= a @ b + C
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    nvcuda::wmma::store_matrix_sync(&out[tile_row_index_in_out * 16 * mat2_width + tile_col_index_in_out], frag_c, mat2_width, nvcuda::wmma::mem_row_major);
}

template<class T1, class T2, typename _ = std::enable_if_t<std::is_same_v<T2, half>>>
std::vector<T1> matmul_with_tensor_core(
    T2 const *mat1, T2 const *mat2, 
    std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width
) {
    T1 *device_out;
    cudaMalloc(&device_out, sizeof(T1) * mat1_height * mat2_width);
    T2 *device_mat1;
    cudaMalloc(&device_mat1, sizeof(T2) * mat1_height * mat1_width);
    cudaMemcpy(device_mat1, mat1, sizeof(T2) * mat1_height * mat1_width, cudaMemcpyDefault);
    T2 *device_mat2;
    cudaMalloc(&device_mat2, sizeof(T2) * mat1_width * mat2_width);
    cudaMemcpy(device_mat2, mat2, sizeof(T2) * mat1_width * mat2_width, cudaMemcpyDefault);

    constexpr int warp_size = 32;
    dim3 threads = {256, 1, 1};
    const int num_tiles_in_out = (mat1_height * mat2_width) / (16 * 16);
    dim3 blocks = {std::max(num_tiles_in_out / (threads.x / warp_size), 1u), 1, 1};

    kernel_matmul_tc<T1, T2><<<blocks, threads>>>(
        device_out, device_mat1, device_mat2, mat1_height, mat1_width, mat2_width
    );

    std::vector<T1> out(mat1_height * mat2_width, 0);
    cudaMemcpy(out.data(), device_out, sizeof(T1) * mat1_height * mat2_width, cudaMemcpyDefault);
    cudaFree(device_out);
    cudaFree(device_mat1);
    cudaFree(device_mat2);

    return out;
}

template<class T1, class T2, typename _ = std::enable_if_t<std::is_same_v<T2, half>>>
std::vector<T1> matmul_with_tensor_core_shm(
    T2 const *mat1, T2 const *mat2, 
    std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width
) {
    T1 *device_out;
    cudaMalloc(&device_out, sizeof(T1) * mat1_height * mat2_width);
    T2 *device_mat1;
    cudaMalloc(&device_mat1, sizeof(T2) * mat1_height * mat1_width);
    cudaMemcpy(device_mat1, mat1, sizeof(T2) * mat1_height * mat1_width, cudaMemcpyDefault);
    T2 *device_mat2;
    cudaMalloc(&device_mat2, sizeof(T2) * mat1_width * mat2_width);
    cudaMemcpy(device_mat2, mat2, sizeof(T2) * mat1_width * mat2_width, cudaMemcpyDefault);

    constexpr int warp_size = 32;
    dim3 threads = {256, 1, 1};
    const int num_tiles_in_out = (mat1_height * mat2_width) / (16 * 16);
    dim3 blocks = {std::max(num_tiles_in_out / (threads.x / warp_size), 1u), 1, 1};

    kernel_matmul_tc_shm<T1, T2><<<blocks, threads>>>(
        device_out, device_mat1, device_mat2, mat1_height, mat1_width, mat2_width
    );

    std::vector<T1> out(mat1_height * mat2_width, 0);
    cudaMemcpy(out.data(), device_out, sizeof(T1) * mat1_height * mat2_width, cudaMemcpyDefault);
    cudaFree(device_out);
    cudaFree(device_mat1);
    cudaFree(device_mat2);

    return out;
}

} // namespace matrix_multiply

} // namespace combat