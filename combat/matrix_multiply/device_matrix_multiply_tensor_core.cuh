#include "mma.h"

template<class T>
__global__ void kernel_matmul_tc(
    T *out, T const* mat1, T const *mat2,
    std::size_t mat1_height, std::size_t matl_width,std::size_t mat2_width
) {
    //假设out可以被16x16的Tile完全分块
    //每个Warp计算出一个out的Tile
    constexpr int tile_size = 16;
    constexpr int warp_size = 32;
    int const warp_index = (threadIdx.x + blockDim.x* blockIdx.x) / warp_size;

    int const num_tiles_per_row_in_out = mat2_width / tile_size;
    int const tile_row_index_in_out = warp_index / num_tiles_per_row_in_out;
    int const tile_col_index_in_out = warp_index % num_tiles_per_row_in_out;

    int const num_tiles_per_row_in_mat1 = mat1_width/ tile_size;
    int shift_to_mat1_first_row_tile = tile_row_index_in_out * tile_size * mat1_width;
    int shift_to_mat2_first_col_tile = tile_col_index_in_out * tile_size;

    nvcuda::wmma::fragment<matrix_a, 16, 16, 16, T,row_order> frag_a;
    nvcuda::wmma::fragment<matrix_b, 16, 16, 16, T,row_order> frag_b;
    nvcuda::wmma::fragment<accumulator, 16, 16, 16, T> frag_c:
    nvcuda::wmma::fill_fragment(frag_c, 0);

    for (int i=0; i<num_tiles_per_row_in_mat1; ++i) {
        int shift_to_mat1 = shift_to_mat1_first_row_tile+ i * tile_size;
        int shift_to_mat2 = shift_to_mat2_first_col_tile + i * tile_size * mat2_width;
        nvcuda::wmma::load_matrix_sync(frag_a, &mat1[shift_to_mati], mat1_width);
        nvcuda::wmma::load_matrix_sync(frag_b, &mat2[shift_to_mat2], mat2_width);
        // C= a @ b + C
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, row_order);
    }
    store_matrix_sync(&out[tile_row_index_in_out * 16 * mat2_width + tile_col_index_in_out], frag_c, mat2_width, row_order);
}

template<class T>
__global__ void kernel_matmul_tc_shared(
    T *out, T const* mat1, T const *mat2,
    std::size_t mat1_height, std::size_t matl_width,std::size_t mat2_width
) {
    /*
        保证调用时是 <<<..., 256>>>
        使用共享内存, 每个 block 可以共享 A 的 tile
        需要保证每个 block 都处理 out 里同一行的 tile
        
        比如 out: [64, 32], blockDim.x = 256 = 8个线程束
        前2个处理 out 的 [:16, :]
        如果我们往共享内存里保存了 sm[16][16] = out[:16,:16], 然后同步
        实际上这个 sm 只有前两个束计算的应该用到
    */
    constexpr int tile_size = 16;
    __shared__ T shared_a_tile[tile_size][tile_size];

    constexpr int warp_size = 32;
    int const warp_index = (threadIdx.x + blockDim.x* blockIdx.x) / warp_size;

    int const num_tiles_per_row_in_out = mat2_width / tile_size;
    if (num_tiles_per_row_in_out < 8 || num_tiles_per_row_in_out % 8 != 0) {
        // 保证 blockDim.x == 256 的情况下, 如果每行不是 8 整数倍的 Tile 数, 说明不合规
        return;
    }

    int const tile_row_index_in_out = warp_index / num_tiles_per_row_in_out;
    int const tile_col_index_in_out = warp_index % num_tiles_per_row_in_out;

    int const num_tiles_per_row_in_mat1 = mat1_width / tile_size;
    int shift_to_mat1_first_row_tile = tile_row_index_in_out * tile_size * mat1_width;
    int shift_to_mat2_first_col_tile = tile_col_index_in_out * tile_size;

    nvcuda::wmma::fragment<matrix_a, 16, 16, 16, T, row_order> frag_a;
    nvcuda::wmma::fragment<matrix_b, 16, 16, 16, T, row_order> frag_b;
    nvcuda::wmma::fragment<accumulator, 16, 16, 16, T> frag_c:
    nvcuda::wmma::fill_fragment(frag_c, 0);

    for (int i=0; i<num_tiles_per_row_in_mat1; ++i) {
        int shift_to_mat1 = shift_to_mat1_first_row_tile+ i * tile_size;
        int sy = threadIdx.x/tile_size;
        int sx = threadIdx.x%tile_size;
        shared_a_tile[sy][sx] = mat1[shift_to_mat1+sy*mat1_width+sx];
        __syncthreads();

        int shift_to_mat2 = shift_to_mat2_first_col_tile + i * tile_size * mat2_width;
        nvcuda::wmma::load_matrix_sync(frag_a, &shared_a_tile[0][0], tile_size);
        nvcuda::wmma::load_matrix_sync(frag_b, &mat2[shift_to_mat2], mat2_width);
        // C= a @ b + C
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, row_order);
    }

    store_matrix_sync(&out[tile_row_index_in_out * 16 * mat2_width + tile_col_index_in_out], frag_c, mat2_width, row_order);
}

template<class
std::vector<T> matmul_tc(
    T const *mat1, T const *mat2, 
    std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width
) {
    T *device_out;
    cudaMalloc(&device_out, sizeof(T) * mat1_height * mat2_width);
    T *device_mat1;
    cudaMalloc(&device_mat1, sizeof(T) * mat1_height * mat1_width);
    cudaMemcpy(device_mat1, mat1, sizeof(T) * mat1_height * mat1_width, cudaMemcpyDefault);
    T *device_mat2;
    cudaMalloc(&device_mat2, sizeof(T) * mat1_width * mat2_width);
    cudaMemcpy(device_mat2, mat2, sizeof(T) * mat1_width * mat2_width, cudaMemcpyDefault);

    constexpr int warp_size = 32;
    dim3 threads = {256, 1, 1};
    const int num_tiles_in_out = (mat1_height * mat2_width) / (16 * 16);
    dim3 blocks = {num_tiles_in_out / (threads.x / warp_size), 1, 1};

    kernel_matrix_multiply_tiling<T, TileSize><<<blocks, threads>>>(
        device_out, device_mat1, device_mat2, mat1_height, mat1_width, mat2_width
    );

    std::vector<T> out(mat1_height * mat2_width, 0);
    cudaMemcpy(out.data(), device_out, sizeof(T) * mat1_height * mat2_width, cudaMemcpyDefault);
    cudaFree(device_out);
    cudaFree(device_mat1);
    cudaFree(device_mat2);

    return out;
}