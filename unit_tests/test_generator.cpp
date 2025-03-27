#include "tools/timer.hpp"
#include "tools/generator.hpp"
#include "tools/show.hpp"

#include <thread>
#include <chrono>

using namespace combat::tools;

int main() {
    {
        auto t = Timer("generate random matrix with seed");
        auto matrix = generate_random_matrix<int>(8, 8, 0, 1, 10);
        show_matrix(matrix, 8, "8x8 matrix");
    }

    {
        auto t = Timer("generate random matrix without seed");
        auto matrix = generate_random_matrix<float>(256, 4, 0, 1);
        show_matrix_first_row(matrix, 4, "256x4 matrix");
    }

    {
        auto t = Timer("generate random matrix without seed");
        auto matrix = generate_random_matrix<double>(4, 256, 0, 1);
        show_matrix_first_column(matrix, 256, "4x256 matrix");
    }

    {
        auto t = Timer("generate ones matrix");
        auto matrix = generate_ones_matrix<int>(4, 4);
        show_matrix(matrix, 4, "4x4 matrix");
    }

    {
        auto t = Timer("generate zeros matrix");
        auto matrix = generate_zeros_matrix<float>(5, 5);
        show_matrix(matrix, 5, "5x5 matrix");
    }

    {
        auto t = Timer("generate ladder matrix");
        auto matrix = generate_ladder_matrix<double>(5, 5);
        show_matrix(matrix, 5, "5x5 matrix");
    }

    {
        auto t = Timer("generate random sequence");
        auto sequence = generate_random_sequence<double>(15, 0, 1, 0);
        show_sequence(sequence, 5, "len=15 sequence");
    }

    {
        auto t = Timer("generate constant sequence");
        auto sequence = generate_constant_sequence<int>(15, 5);
        show_sequence(sequence);
    }

    return 0;
}
