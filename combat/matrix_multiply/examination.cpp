#include "combat/matrix_multiply/host_matrix_multiply.hpp"

#include "tools/generator.hpp"
#include "tools/show.hpp"
#include "tools/timer.hpp"

namespace combat {

namespace matrix_multiply {

template<class T>
void check_result(T* ground_truth, T* calculate_result, std::size_t height, std::size_t width, char const *tag) {
    static_assert(std::is_same_v<T, int>, "现在只支持 int 类型的比较, 浮点类型需要考虑误差");

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
        if (num_elements < 100) {
            tools::show_matrix(ground_truth, width, "ground truth");
            tools::show_matrix(calculate_result, width, "calculate result");
        }
    }
}

int main() {
    // generate matrix
    std::size_t constexpr mat1_height = 1 << 10;
    std::size_t constexpr mat1_width  = 1 << 10;
    std::size_t constexpr mat2_height = mat1_width;
    std::size_t constexpr mat2_width  = 1 << 10;
    std::size_t constexpr out_height = mat1_height;
    std::size_t constexpr out_width = mat2_width;
    std::size_t constexpr num_out_element = out_height * out_width;

    auto mat1 = tools::generate_ladder_matrix<int>(mat2_height, mat2_width);
    auto mat2 = tools::generate_ones_matrix<int>(mat1_height, mat1_width);
    // mat1: [[0, 0, ..., 0], [1, 1, ..., 1], ... , [n-1, n-1, ..., n-1]]
    // mat2: all 1
    // let S=sum(1,2,...n) out should be (if both squared matrix)
    // [[0, 0, ..., 0], [n, n, ..., n], ..., [(n-1)n, (n-1)n, ..., (n-1)n]]
    std::vector<int> ground_truth;
    ground_truth.reserve(mat1_height*mat2_width);
    for (std::size_t i=0; i<mat1_height; ++i)
        for (std::size_t j=0; j<mat2_width; ++j)
            ground_truth.push_back(i*mat1_width);

    // quiz 1: calculate in host cpu
    {
        auto const *tag = "matrix multiply host";
        std::vector<int> out;
        out.reserve(num_out_element);
        {
            auto t = tools::Timer(tag);
            matrix_multiply_host<int>(out.data(), mat1.data(), mat2.data(), mat1_height, mat1_width, mat2_width);
        }
        check_result(ground_truth.data(), out.data(), out_height, out_width, tag);
    }

    return 0;
}

} // namespace matrix_multiply

} // namespace combat