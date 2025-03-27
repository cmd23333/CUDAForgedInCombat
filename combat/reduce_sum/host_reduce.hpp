#pragma once

#include "omp.h"

namespace combat {

namespace reduce_sum {

template<class T>
T reduce_sum_host(T const *data, std::size_t length) {
    /*
        提示: 使用多核并行
    */
    T sum = 0;
    #pragma omp parallel for reduction(+ : sum) 
    for (std::size_t i=0; i<length; ++i) {
        sum += data[i];
    }

    return sum;
}

} // namespace reduce_sum

} // namespace combat