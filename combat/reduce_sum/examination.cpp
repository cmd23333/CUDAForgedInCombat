#include "combat/reduce_sum/host_reduce.hpp"

#include "tools/generator.hpp"
#include "tools/show.hpp"
#include "tools/timer.hpp"

template<class T>
void check_result(T ground_truth, T calculate_result, char const *tag) {
    if (ground_truth == calculate_result)
        std::cout << "Pass " << tag << std::endl;
    else
        std::cout << "Fail " << tag << ", with ground_truth = " << ground_truth 
                  << ", but calculate_result = " << calculate_result << std::endl; 
}

int main() {
    // generate sequence
    std::size_t length = 1 << 24;
    auto sequence = combat::tools::generate_constant_sequence<int>(length, 1);
    int ground_truth = length;

    // quiz 1: calculate in host cpu, using multi-core
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum host (100 times)");
        for (int i=0; i<100; ++i)
            result = combat::reduce_sum::reduce_sum_host(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum host");
    }

    return 0;
}