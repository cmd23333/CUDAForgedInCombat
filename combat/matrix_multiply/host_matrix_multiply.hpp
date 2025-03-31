#pragma once
#include <stdlib.h>
#include <vector>

namespace combat {

namespace matrix_multiply {

template<class T>
std::vector<T> matrix_multiply_host(T const* mat1, T const *mat2, std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width) {
    /*
        assume mat1_width == mat2_height

        提示: 选择最高效的循环顺序
    */
    auto out = std::vector<T>(mat1_height*mat2_width, 0);

    for (std::size_t i=0; i<mat1_height; ++i) {
        for (std::size_t k=0; k<mat1_width; ++k) {
            const auto t = mat1[i * mat1_width + k];
            for (std::size_t j=0; j<mat2_width; ++j) {
                out[i * mat2_width + j] += t * mat2[k * mat2_width + j];
            }
        }
    }

    return out;
}

} // namespace matrix_multiply

} // namespace combat