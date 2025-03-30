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
    1.12 reduce_with_tensor_core（提示：全 1 的矩阵可以拿来做求和）
*/

template <class T>
__global__ void kernel_reduce_with_tensor_core(T *sum, T const *data, std::size_t length) {
    /*
        全 1 的矩阵可以拿来做求和
    */
}

template <class T>
T reduce_with_tensor_core(T const *host_data, std::size_t length) {
    T sum = 0;
    return sum;
}

} // namespace reduce_sum

} // namespace combat
