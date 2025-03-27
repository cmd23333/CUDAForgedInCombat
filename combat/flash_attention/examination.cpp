#include "combat/flash_attention/host_flash_attention.hpp"

#include "tools/generator.hpp"
#include "tools/show.hpp"
#include "tools/timer.hpp"

namespace combat {

namespace flash_attention {

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
    std::size_t constexpr batch_size = 2;
    std::size_t constexpr num_heads  = 4;
    std::size_t constexpr depth = 128;
    std::size_t constexpr seq_len  = 32;

    auto Q = tools::generate_batch_tensor<int>(batch_size, num_heads, seq_len, depth);
    auto K = tools::generate_batch_tensor<int>(batch_size, num_heads, seq_len, depth);
    auto V = tools::generate_batch_tensor<int>(batch_size, num_heads, seq_len, depth);
    // TODO: 生成 GT
    std::vector<int> ground_truth;

    // quiz 1: calculate in host cpu
    {
        auto const *tag = "flash attention host";
        std::vector<int> out;
        out.reserve(Q.size());
        {
            auto t = tools::Timer(tag);
            flash_attention_host<int>(out.data(), Q.data(), K.data(), V.data(), batch_size, num_heads, seq_len, depth);
        }
    }

    return 0;
}

} // namespace flash_attention

} // namespace combat