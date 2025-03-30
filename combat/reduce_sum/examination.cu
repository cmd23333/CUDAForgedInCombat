#include "combat/reduce_sum/host_reduce.hpp"
#include "combat/reduce_sum/device_reduce_basic.cuh"
#include "combat/reduce_sum/device_reduce_cg.cuh"

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
    std::size_t length = 1 << 28;
    auto sequence = combat::tools::generate_constant_sequence<int>(length, 1);
    int ground_truth = length;

    // quiz 1: calculate in host cpu, using multi-core
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum host (10 times)");
        for (int i=0; i<100; ++i)
            result = combat::reduce_sum::reduce_sum_host(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum host");
    }

    // quiz 2: calculate in cuda, basic reduce
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda basic version (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v0(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda basic version");
    }

    // quiz 3: calculate in cuda, reduce with grid stride loop
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda with grid stride loop (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v1(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda with grid stride loop");
    }

    // quiz 4: calculate in cuda, reduce with grid stride loop, shared memory to store threads sum
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda with grid stride loop + shared memory (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v2(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda with grid stride loop + shared memory");
    }


    // quiz 5: calculate in cuda, same as v2, but change blocks 256 -> 288
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda with grid stride loop + shared memory + block 288 (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v3(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda with grid stride loop + shared memory + block 288");
    }

    // quiz 6: calculate in cuda, unroll serial reduce sum, using __syncwarp()
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda with unroll serial reduce sum, using __syncwarp() (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v4(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda with unroll serial reduce sum, using __syncwarp()");
    }

    // quiz 7: calculate in cuda, unroll serial reduce sum, using __syncwarp()
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda with unroll serial reduce sum, aggressively using __syncwarp() (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v5(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda with unroll serial reduce sum, aggressively using __syncwarp()");
    }

    // quiz 8: calculate in cuda, using cg and shlf_down
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda, using cg and shlf_down (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v6(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda, using cg and shlf_down");
    }

    // quiz 9: calculate in cuda, warp-only
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda, warp-only (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v7(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda, warp-only");
    }

    // quiz 10: calculate in cuda, warp-only + cg::reduce
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda, warp-only + cg::reduce (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v8(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda, warp-only + cg::reduce");
    }

    // quiz 11: calculate in cuda, warp-only + vector loading
    {
        int result = 0;
        auto t = combat::tools::Timer("reduce sum cuda, warp-only + vector loading (10 times)");
        for (int i=0; i<10; ++i)
            result = combat::reduce_sum::reduce_sum_device_v7_vl(sequence.data(), length);
        check_result(ground_truth, result, "reduce sum cuda, warp-only + vector loading");
    }
    
    return 0;
}