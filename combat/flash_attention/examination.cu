#include "combat/flash_attention/host_multi_head_attention.hpp"
#include "combat/flash_attention/host_flash_attention.hpp"
#include "combat/flash_attention/device_flash_attention_unreal.cuh"

#include "tools/generator.hpp"
#include "tools/show.hpp"
#include "tools/timer.hpp"

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
        if (num_elements < 8*16+1) {
            combat::tools::show_matrix(ground_truth, height, width, "ground truth");
            combat::tools::show_matrix(calculate_result, height, width, "calculate result");
        }
    }
}

int main() {
    // generate matrix
    std::size_t constexpr batch_size = 1;
    std::size_t constexpr num_heads  = 1;
    std::size_t constexpr depth = 16;
    std::size_t constexpr seq_len  = 8;

    int const seed = 233;
    auto Q = combat::tools::generate_batch_tensor<float>(batch_size, num_heads, seq_len, depth, 0, 1, seed);
    auto K = combat::tools::generate_batch_tensor<float>(batch_size, num_heads, seq_len, depth, 0, 1, seed+1);
    auto V = combat::tools::generate_batch_tensor<float>(batch_size, num_heads, seq_len, depth, 0, 1, seed+2);

    std::vector<float> ground_truth;
    ground_truth.reserve(V.size());
    combat::flash_attention::multi_head_attention_host(
        ground_truth.data(), Q.data(), K.data(), V.data(), 
        batch_size, num_heads, seq_len, depth
    );

    combat::tools::show_matrix(Q.data(), seq_len, depth, "Q");
    combat::tools::show_matrix(K.data(), seq_len, depth, "K");
    combat::tools::show_matrix(V.data(), seq_len, depth, "V");
    combat::tools::show_matrix(ground_truth.data(), seq_len, depth, "mha");

    // quiz 1: calculate in host cpu
    if (0) {
        auto const *tag = "flash attention host";
        std::vector<float> out;
        out.reserve(Q.size());
        {
            auto t = combat::tools::Timer(tag);
            combat::flash_attention::flash_attention_host<float>(out.data(), Q.data(), K.data(), V.data(), batch_size, num_heads, seq_len, depth);
        }
        check_result(ground_truth.data(), out.data(), seq_len, depth, "flash attention host");
    }

    // quiz 2: calculate in device cuda
    {
        auto const *tag = "flash attention cuda";
        std::vector<float> out;
        out.reserve(Q.size());
        {
            auto t = combat::tools::Timer(tag);
            combat::flash_attention::flash_attention_cuda<float, depth>(out.data(), Q.data(), K.data(), V.data(), batch_size, num_heads, seq_len);
        }
        check_result(ground_truth.data(), out.data(), seq_len, depth, "flash attention cuda");
    }

    return 0;
}
