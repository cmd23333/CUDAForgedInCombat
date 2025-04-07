#include "combat/matrix_multiply/host_matrix_multiply.hpp"
#include "combat/matrix_multiply/device_matrix_multiply_basic.cuh"
#include "combat/matrix_multiply/device_matrix_multiply_tensor_core.cuh"

#include "tools/generator.hpp"
#include "tools/show.hpp"
#include "tools/timer.hpp"

#include "cuda_fp16.h"

namespace combat {

namespace matrix_multiply {

template<class T>
void check_result(T* ground_truth, T* calculate_result, std::size_t height, std::size_t width, char const *tag) {
    auto num_elements = height * width;
    bool passed = true;
    for (std::size_t i=0; i<num_elements; ++i) {
        if (ground_truth[i] != calculate_result[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Pass " << tag << std::endl;
    } else {
        std::cout << "Fail " << tag << std::endl;
        if (num_elements < 257) {
            tools::show_matrix(ground_truth, height, width, "ground truth");
            tools::show_matrix(calculate_result, height, width, "calculate result");
        }
    }
}

} // namespace matrix_multiply

} // namespace combat

int main() {
    // generate matrix
    std::size_t constexpr mat1_height = 1024;
    std::size_t constexpr mat1_width  = 1024;
    std::size_t constexpr mat2_height = mat1_width;
    std::size_t constexpr mat2_width  = 1024;
    std::size_t constexpr out_height = mat1_height;
    std::size_t constexpr out_width = mat2_width;
    std::size_t constexpr num_out_element = out_height * out_width;

    auto mat1 = combat::tools::generate_ladder_matrix<float>(mat1_height, mat1_width);
    auto mat2 = combat::tools::generate_ones_matrix<float>(mat2_height, mat2_width);
    // mat1: [[0, 0, ..., 0], [1, 1, ..., 1], ... , [n-1, n-1, ..., n-1]]
    // mat2: all 1
    // let S=sum(1,2,...n) out should be (if both squared matrix)
    // [[0, 0, ..., 0], [n, n, ..., n], ..., [(n-1)n, (n-1)n, ..., (n-1)n]]
    std::vector<float> ground_truth;
    ground_truth.reserve(mat1_height*mat2_width);
    for (std::size_t i=0; i<mat1_height; ++i)
        for (std::size_t j=0; j<mat2_width; ++j)
            ground_truth.push_back(i*mat1_width);

    // quiz 1: calculate in host cpu
    {
        auto const *tag = "matrix multiply host";

        auto t = combat::tools::Timer(tag);
        auto out = combat::matrix_multiply::matrix_multiply_host<float>(mat1.data(), mat2.data(), mat1_height, mat1_width, mat2_width);
        combat::matrix_multiply::check_result(ground_truth.data(), out.data(), out_height, out_width, tag);
    }

    // quiz 2: calculate in device cuda trival
    {
        auto const *tag = "matrix multiply cuda trival";

        auto t = combat::tools::Timer(tag);
        auto out = combat::matrix_multiply::matrix_multiply_trival<float>(mat1.data(), mat2.data(), mat1_height, mat1_width, mat2_width);
        combat::matrix_multiply::check_result(ground_truth.data(), out.data(), out_height, out_width, tag);
    }

    // quiz 3: calculate in device cuda tiling
    {
        auto const *tag = "matrix multiply cuda tiling";

        auto t = combat::tools::Timer(tag);
        auto out = combat::matrix_multiply::matrix_multiply_tiling<float>(mat1.data(), mat2.data(), mat1_height, mat1_width, mat2_width);
        combat::matrix_multiply::check_result(ground_truth.data(), out.data(), out_height, out_width, tag);
    }

    // quiz 4: calculate in device cuda tensor core
    {
        auto const *tag = "matrix multiply cuda tensor core";
        auto mat1 = combat::tools::generate_ladder_matrix<half>(mat1_height, mat1_width);
        auto mat2 = combat::tools::generate_ones_matrix<half>(mat2_height, mat2_width);
        auto t = combat::tools::Timer(tag);
        auto out = combat::matrix_multiply::matmul_with_tensor_core<float, half>(mat1.data(), mat2.data(), mat1_height, mat1_width, mat2_width);
        combat::matrix_multiply::check_result(ground_truth.data(), out.data(), out_height, out_width, tag);
    }

    // quiz 5: calculate in device cuda tensor core shm
    {
        auto const *tag = "matrix multiply cuda tensor core shm";
        auto mat1 = combat::tools::generate_ladder_matrix<half>(mat1_height, mat1_width);
        auto mat2 = combat::tools::generate_ones_matrix<half>(mat2_height, mat2_width);
        auto t = combat::tools::Timer(tag);
        auto out = combat::matrix_multiply::matmul_with_tensor_core_shm<float, half>(mat1.data(), mat2.data(), mat1_height, mat1_width, mat2_width);
        combat::matrix_multiply::check_result(ground_truth.data(), out.data(), out_height, out_width, tag);
    }

    return 0;
}