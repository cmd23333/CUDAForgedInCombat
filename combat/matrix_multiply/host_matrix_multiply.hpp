#pragma once

namespace combat {

namespace matrix_multiply {

template<class T>
void matrix_multiply_host(T *out, T const* mat1, T const *mat2, std::size_t mat1_height, std::size_t mat1_width, std::size_t mat2_width) {
    /*
        assume mat1_width == mat2_height

        提示: 选择最高效的循环顺序
    */

}

} // namespace matrix_multiply

} // namespace combat