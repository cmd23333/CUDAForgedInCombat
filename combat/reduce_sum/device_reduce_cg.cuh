#pragma once

#include "stdlib.h"
#include "omp.h"

#include <iostream>
#include <chrono>
#include <numeric>

#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"
#include "tools/math.hpp"

namespace cg = cooperative_groups;

namespace combat {

namespace reduce_sum {
/*
    1.8 reduce_sum6（提示：用 cooperative_group 做同步，最后串行求和用 shfl_down）
    1.9 reduce_sum7（提示：不再使用共享内存，最后用 AtomicAdd，为什么要求 threads 是 32 的倍数？）
    1.10 reduce_sum8（提示：用 cooperative_group::reduce）
    1.11 reduce_sum7v（提示：用 float4 一次读取 4 个数字做加法，L1 的 Cache Line 大小是多少？）
    1.12 reduce_with_tensor_core（提示：全 1 的矩阵可以拿来做求和）
*/

template <class T>
__global__ void kernel_reduce_sum_v6(T *sum, T const *data, std::size_t length) {
    /*
        用 cooperative_group 做同步, 最后求和用 shfl_down
    */
    extern __shared__ T thread_sums[];
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int thread_rank_in_block = block.thread_rank();
    thread_sums[thread_rank_in_block] = 0;

    int thread_rank_in_grid = grid.thread_rank();
    int stride = grid.size();

    T local_sum = 0;
    for (int temp_id=thread_rank_in_grid; temp_id<length; temp_id+=stride) {
        local_sum += data[temp_id];
    }
    thread_sums[thread_rank_in_block] = local_sum;
    block.sync();   // 代替 __syncthreads();

    for (int index=thread_rank_in_block + 256; index<block.size(); index += 256)
        thread_sums[thread_rank_in_block] += thread_sums[index];
    block.sync();

    if (thread_rank_in_block < 128 && thread_rank_in_block + 128 < block.size())
        thread_sums[thread_rank_in_block] += thread_sums[thread_rank_in_block + 128];
    block.sync();

    if (thread_rank_in_block < 64 && thread_rank_in_block + 64 < block.size())
        thread_sums[thread_rank_in_block] += thread_sums[thread_rank_in_block + 64];
    block.sync();

    if (thread_rank_in_block < 32 && thread_rank_in_block + 32 < block.size())
        thread_sums[thread_rank_in_block] += thread_sums[thread_rank_in_block + 32];
    warp.sync();

    if (warp.meta_group_rank() == 0) {
        thread_sums[thread_rank_in_block] += warp.shfl_down(thread_sums[thread_rank_in_block], 16);
        thread_sums[thread_rank_in_block] += warp.shfl_down(thread_sums[thread_rank_in_block], 8);
        thread_sums[thread_rank_in_block] += warp.shfl_down(thread_sums[thread_rank_in_block], 4);
        thread_sums[thread_rank_in_block] += warp.shfl_down(thread_sums[thread_rank_in_block], 2);
        thread_sums[thread_rank_in_block] += warp.shfl_down(thread_sums[thread_rank_in_block], 1);
        if (thread_rank_in_block == 0)
            sum[block.group_index().x] = thread_sums[thread_rank_in_block];
    }
}

template <class T>
T reduce_sum_device_v6(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 256;
    cudaMalloc(&device_sum, sizeof(T)*blocks);

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v6<<<blocks, threads, sizeof(T)*threads, nullptr>>>(device_sum, device_data, length);
        kernel_reduce_sum_v6<<<1, blocks, sizeof(T)*blocks, nullptr>>>(device_sum, device_sum, blocks);

        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v6 10 times, cost " << sum << " micro-seconds" << std::endl;
        }
    }

    T sum = 0;
    // 这里自动同步了
    cudaMemcpy(&sum, &device_sum[0], sizeof(T), cudaMemcpyDefault);
    cudaFree(device_data);
    cudaFree(device_sum);

    return sum;
}

template <class T>
__global__ void kernel_reduce_sum_v7(T *sum, T const *data, std::size_t length) {
    /*
        WARP-ONLY, 不再使用共享内存. 
        每个线程的部分和存在自己的 local-variable 里.
        最后用 AtomicAdd, 把每一 warp 的结果加到 sum[block.group_index().x] 里
        为什么要求 threads 是 32 的倍数? 否则不能分成完整的 warp
    */

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int thread_rank_in_grid = grid.thread_rank();
    int stride = grid.size();

    T local_sum = 0;
    for (int temp_id=thread_rank_in_grid; temp_id<length; temp_id+=stride) {
        local_sum += data[temp_id];
    }
    warp.sync();   // warp only, 不再 __syncthreads();

    // 每个 warp 算求和
    local_sum += warp.shfl_down(local_sum, 16);
    local_sum += warp.shfl_down(local_sum, 8);
    local_sum += warp.shfl_down(local_sum, 4);
    local_sum += warp.shfl_down(local_sum, 2);
    local_sum += warp.shfl_down(local_sum, 1);

    // 每个 warp 的第一个线程做求和
    if (warp.thread_rank() == 0)
        atomicAdd(&sum[block.group_index().x], local_sum);
}

template <class T>
T reduce_sum_device_v7(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum, *device_sum_1;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 256;
    cudaMalloc(&device_sum, sizeof(T)*blocks);  // 每个 block 存一个
    cudaMalloc(&device_sum_1, sizeof(T));

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v7<<<blocks, threads>>>(device_sum, device_data, length);
        // 注意这里要再开辟一个, 否则会算出的结果会多一份这时的 device_sum[0]

        kernel_reduce_sum_v7<<<1, blocks>>>(device_sum_1, device_sum, blocks);
        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v7 10 times, cost " << sum << " micro-seconds" << std::endl;
        }
    }

    T sum = 0;
    // 这里自动同步了
    cudaMemcpy(&sum, device_sum_1, sizeof(T), cudaMemcpyDefault);
    cudaFree(device_data);
    cudaFree(device_sum);
    cudaFree(device_sum_1);

    return sum;
}

template <class T>
__global__ void kernel_reduce_sum_v8(T *sum, T const *data, std::size_t length) {
    /*
        WARP-ONLY, 不再使用共享内存. 
        每个线程的部分和存在自己的 local-variable 里.
        最后用 cg::reduce 代替 shfl_down
    */

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int thread_rank_in_grid = grid.thread_rank();
    int stride = grid.size();

    T local_sum = 0;
    for (int temp_id=thread_rank_in_grid; temp_id<length; temp_id+=stride) {
        local_sum += data[temp_id];
    }
    warp.sync();   // warp only, 不再 __syncthreads();
    
    // 每个 warp 算求和
    local_sum = cg::reduce(warp, local_sum, cg::plus<T>());

    // 每个 warp 的第一个线程做求和
    if (warp.thread_rank() == 0)
        atomicAdd(&sum[block.group_index().x], local_sum);
}

template <class T>
T reduce_sum_device_v8(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum, *device_sum_1;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 256;
    cudaMalloc(&device_sum, sizeof(T)*blocks);  // 每个 block 存一个
    cudaMalloc(&device_sum_1, sizeof(T));

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v8<<<blocks, threads>>>(device_sum, device_data, length);
        // 注意这里要再开辟一个, 否则会算出的结果会多一份这时的 device_sum[0]
        kernel_reduce_sum_v8<<<1, blocks>>>(device_sum_1, device_sum, blocks);

        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v8 10 times, cost " << sum << " micro-seconds" << std::endl;
        }
    }

    T sum = 0;
    // 这里自动同步了
    cudaMemcpy(&sum, device_sum_1, sizeof(T), cudaMemcpyDefault);
    cudaFree(device_data);
    cudaFree(device_sum);
    cudaFree(device_sum_1);

    return sum;
}

template <class T>
__global__ void kernel_reduce_sum_v7_vl(T *sum, T const *data, std::size_t length) {
    /*
        在 v7 的基础上, 使用 float4 一次 load 4 个数字进来
    */
    static_assert(std::is_same_v<T, int>);

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int thread_rank_in_grid = grid.thread_rank();
    int stride = grid.size();

    int4 local_sum_vl = {0, 0, 0, 0};
    // 如果是 float -> float4
    for (int temp_id=thread_rank_in_grid; temp_id<length / 4; temp_id+=stride) {
        local_sum_vl += reinterpret_cast<int4 const*>(data)[temp_id];
    }
    warp.sync();   // warp only, 不再 __syncthreads();

    T local_sum = local_sum_vl.w + local_sum_vl.x + local_sum_vl.y + local_sum_vl.z;
    // 每个 warp 算求和
    local_sum += warp.shfl_down(local_sum, 16);
    local_sum += warp.shfl_down(local_sum, 8);
    local_sum += warp.shfl_down(local_sum, 4);
    local_sum += warp.shfl_down(local_sum, 2);
    local_sum += warp.shfl_down(local_sum, 1);

    // 每个 warp 的第一个线程做求和
    if (warp.thread_rank() == 0)
        atomicAdd(&sum[block.group_index().x], local_sum);
}

template <class T>
T reduce_sum_device_v7_vl(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum, *device_sum_1;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 256;
    cudaMalloc(&device_sum, sizeof(T)*blocks);  // 每个 block 存一个
    cudaMalloc(&device_sum_1, sizeof(T));

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v7<<<blocks, threads>>>(device_sum, device_data, length);
        // 注意这里要再开辟一个, 否则会算出的结果会多一份这时的 device_sum[0]

        kernel_reduce_sum_v7<<<1, blocks>>>(device_sum_1, device_sum, blocks);
        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v7_vl 10 times, cost " << sum << " micro-seconds" << std::endl;
        }
    }

    T sum = 0;
    // 这里自动同步了
    cudaMemcpy(&sum, device_sum_1, sizeof(T), cudaMemcpyDefault);
    cudaFree(device_data);
    cudaFree(device_sum);
    cudaFree(device_sum_1);

    return sum;
}

} // namespace reduce_sum

} // namespace combat
