#pragma once

#include "stdlib.h"
#include "omp.h"

#include <iostream>
#include <chrono>
#include <numeric>

#include "cuda_runtime.h"
#include "tools/math.hpp"

namespace combat {

namespace reduce_sum {
/*
    reduce_sum0（提示：每次把数组后一半的元素加到前一半，最后一个就是求和；为什么主机上 float 精度更低）
    reduce_sum1（提示：跨步循环，每次计算完，前 blocks*threads 个元素保存了部分和，最后一轮是<<<1, 1>>>；为什么步长很大也没关系？）
    reduce_sum2（提示：在上一步的基础上，每个 thread 将它的部分和存到共享内存里，最后串行做求和，结果是每次计算完，用一个新的大小为 block 的数组，每个元素保存了部分和）
    reduce_sum3 （提示：blocks 不是 2 的整数幂，比如 2070 有 36 个 SM，为什么 blocks 的值是 288 要比 256 好，threads=256）
    reduce_sum4（提示：最后串行求和时循环展开，什么时候用 __syncwarp？）
    reduce_sum5（提示：最后串行求和时循环展开，更激进的用 __syncwarp？为什么小于 32 之后的 if 依然不能去掉
*/

template <class T>
__global__ void kernel_reduce_sum_v0(T *data, std::size_t length) {
    /*
        最基础版本: 每次把数组后一半的元素加到前一半，最后一个就是求和.
        这一方法会破坏数组
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (2*index < length) {
        data[index] += data[2*index];
    }
}

template <class T>
T reduce_sum_device_v0(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    {
        auto start_ts = std::chrono::steady_clock::now();

        // 大于等于 Length 的第一个正的 2 的次幂
        auto adjust_length = tools::least_power_of_two_leq_to_number(length);
        adjust_length = adjust_length / 2;
        for (; adjust_length>0; adjust_length /= 2) {
            int threads = std::min(256, adjust_length);
            int blocks = std::max((adjust_length + threads - 1) / threads, 1);
            kernel_reduce_sum_v0<<<blocks, threads>>>(device_data, adjust_length);
        }

        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v0 10 times, cost " << sum << " micro-seconds" << std::endl;
        }
    }

    T sum = 0;
    // 这里自动同步了
    cudaMemcpy(&sum, &device_data[0], sizeof(T), cudaMemcpyDefault);
    cudaFree(device_data);

    return sum;
}


template <class T>
__global__ void kernel_reduce_sum_v1(T *sum, T const *data, std::size_t length) {
    /*
        使用跨步循环. 每次计算完, 前 blocks*threads 个元素保存了部分和.
        最后一轮用 <<<1, 1>>> 调用
        为什么步长很大也没关系? 相邻的线程先取出数据到 L1 cache 了
    */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    T local_sum = 0;
    for (; index<length; index+=stride) {
        local_sum += data[index];
    }

    sum[blockIdx.x * blockDim.x + threadIdx.x] = local_sum;
}

template <class T>
T reduce_sum_device_v1(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 256;
    cudaMalloc(&device_sum, sizeof(T)*blocks*threads);  // 每个线程存一个

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v1<<<blocks, threads>>>(device_sum, device_data, length);
        kernel_reduce_sum_v1<<<1, blocks>>>(device_sum, device_sum, blocks*threads);
        kernel_reduce_sum_v1<<<1, 1>>>(device_sum, device_sum, blocks);

        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v1 10 times, cost " << sum << " micro-seconds" << std::endl;
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
__global__ void kernel_reduce_sum_v2(T *sum, T const *data, std::size_t length) {
    /*
        在上一步的基础上, 使用共享内存保存每个 blocks 里所有 threads 存储的部分和, 最后串行做求和
        结果是每次计算完, 大小为 block 的数组, 每个元素保存了部分和
    */
    extern __shared__ T thread_sums[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    T local_sum = 0;
    for (; index<length; index+=stride) {
        local_sum += data[index];
    }
    thread_sums[threadIdx.x] = local_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        T block_sum = 0;
        for (std::size_t i=0; i<blockDim.x; ++i) {
            block_sum += thread_sums[i];
        }
        sum[blockIdx.x] = block_sum;
    }
}

template <class T>
T reduce_sum_device_v2(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 256;
    cudaMalloc(&device_sum, sizeof(T)*blocks);

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v2<<<blocks, threads, sizeof(T)*threads, nullptr>>>(device_sum, device_data, length);
        kernel_reduce_sum_v2<<<1, blocks, sizeof(T)*blocks, nullptr>>>(device_sum, device_sum, blocks);

        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v2 10 times, cost " << sum << " micro-seconds" << std::endl;
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
T reduce_sum_device_v3(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 288;
    cudaMalloc(&device_sum, sizeof(T)*blocks);

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v2<<<blocks, threads, sizeof(T)*threads, nullptr>>>(device_sum, device_data, length);
        kernel_reduce_sum_v2<<<1, blocks, sizeof(T)*blocks, nullptr>>>(device_sum, device_sum, blocks);

        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v3 10 times, cost " << sum << " micro-seconds" << std::endl;
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
__global__ void kernel_reduce_sum_v4(T *sum, T const *data, std::size_t length) {
    /*
        在上一步的基础上, 使用共享内存保存每个 blocks 里所有 threads 存储的部分和, 最后对共享数组做求和
        求和使用循环展开, 并且使用 __syncthreads() 和 __syncwarp()
    */
    extern __shared__ T thread_sums[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    T local_sum = 0;
    for (; index<length; index+=stride) {
        local_sum += data[index];
    }
    thread_sums[threadIdx.x] = local_sum;
    __syncthreads();

    for (int index=threadIdx.x + 256; index<blockDim.x; index += 256)
        thread_sums[threadIdx.x] += thread_sums[index];
    __syncthreads();

    if (threadIdx.x < 128)
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 128];
    __syncthreads();

    if (threadIdx.x < 64)
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 64];
    __syncthreads();

    if (threadIdx.x < 32)
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 32];
    __syncthreads();

    if (threadIdx.x < 16) 
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 16];
    __syncwarp();

    if (threadIdx.x < 8) 
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 8];
    __syncwarp();

    if (threadIdx.x < 4) 
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 4];
    __syncwarp();

    if (threadIdx.x < 2) 
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 2];
    __syncwarp();

    if (threadIdx.x < 1) {
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 1];
        sum[blockIdx.x] = thread_sums[threadIdx.x];
    }
}

template <class T>
T reduce_sum_device_v4(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 256;
    cudaMalloc(&device_sum, sizeof(T)*blocks);

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v4<<<blocks, threads, sizeof(T)*threads, nullptr>>>(device_sum, device_data, length);
        kernel_reduce_sum_v4<<<1, blocks, sizeof(T)*blocks, nullptr>>>(device_sum, device_sum, blocks);

        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v4 10 times, cost " << sum << " micro-seconds" << std::endl;
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
__global__ void kernel_reduce_sum_v5(T *sum, T const *data, std::size_t length) {
    /*
        使用共享内存保存每个 blocks 里所有 threads 存储的部分和, 最后对共享数组做求和
        求和使用循环展开, 并且使用 __syncthreads() 和 __syncwarp()
        在上一步的基础上, 更激进的用 __syncwarp()
        为什么小于 32 之后的 if 依然不能去掉? 担心 read-after-write 问题
    */
    extern __shared__ T thread_sums[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    T local_sum = 0;
    for (; index<length; index+=stride) {
        local_sum += data[index];
    }
    thread_sums[threadIdx.x] = local_sum;
    __syncthreads();

    for (int index=threadIdx.x + 256; index<blockDim.x; index += 256)
        thread_sums[threadIdx.x] += thread_sums[index];
    __syncthreads();

    if (threadIdx.x < 128)
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 128];
    __syncthreads();

    if (threadIdx.x < 64)
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 64];
    __syncthreads();

    if (threadIdx.x < 32)
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 32];
    __syncwarp();       // 注意, 这里就可以用 __syncwarp 了. 从结果来看反而更慢了?

    if (threadIdx.x < 16) 
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 16];
    __syncwarp();

    if (threadIdx.x < 8) 
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 8];
    __syncwarp();

    if (threadIdx.x < 4) 
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 4];
    __syncwarp();

    if (threadIdx.x < 2) 
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 2];
    __syncwarp();

    if (threadIdx.x < 1) {
        thread_sums[threadIdx.x] += thread_sums[threadIdx.x + 1];
        sum[blockIdx.x] = thread_sums[threadIdx.x];
    }
}

template <class T>
T reduce_sum_device_v5(T const *host_data, std::size_t length) {
    static std::vector<float> workload_timer;

    T *device_data, *device_sum;
    cudaMalloc(&device_data, sizeof(T)*length);
    cudaMemcpy(device_data, host_data, sizeof(T)*length, cudaMemcpyDefault);

    int threads = 256;
    int blocks = 256;
    cudaMalloc(&device_sum, sizeof(T)*blocks);

    {
        auto start_ts = std::chrono::steady_clock::now();

        kernel_reduce_sum_v5<<<blocks, threads, sizeof(T)*threads, nullptr>>>(device_sum, device_data, length);
        kernel_reduce_sum_v5<<<1, blocks, sizeof(T)*blocks, nullptr>>>(device_sum, device_sum, blocks);

        auto end_ts = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts - start_ts).count();
        workload_timer.push_back(duration);
        if (workload_timer.size() == 10) {
            auto sum = std::accumulate(workload_timer.begin(), workload_timer.end(), 0, std::plus<>{});
            std::cout << "Run reduce_sum_device_v5 10 times, cost " << sum << " micro-seconds" << std::endl;
        }
    }

    T sum = 0;
    // 这里自动同步了
    cudaMemcpy(&sum, &device_sum[0], sizeof(T), cudaMemcpyDefault);
    cudaFree(device_data);
    cudaFree(device_sum);

    return sum;
}

} // namespace reduce_sum

} // namespace combat
