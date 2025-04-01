#pragma once
#include <random>
#include <vector>

namespace combat {

namespace tools {

template <class T>
std::vector<T> generate_random_matrix(std::size_t height, std::size_t width, T min=0, T max=1, int seed=-1) {
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }

    auto num_elements = height * width;
    std::vector<T> matrix;
    matrix.reserve(num_elements);

    std::default_random_engine generator{static_cast<unsigned int>(seed)};
    if constexpr (std::is_same_v<T, int>) {
        std::uniform_int_distribution<int> dist(min, max);
        for (std::size_t i=0; i<num_elements; ++i)
            matrix.push_back(dist(generator));

    } else {
        std::uniform_real_distribution<T> dist(min, max);
        for (std::size_t i=0; i<num_elements; ++i)
            matrix.push_back(dist(generator));
    }

    return matrix;
}

template <class T>
std::vector<T> generate_zeros_matrix(std::size_t height, std::size_t width) {
    auto num_elements = height * width;
    std::vector<T> matrix;
    matrix.reserve(num_elements);

    for (std::size_t i=0; i<num_elements; ++i)
        matrix.push_back(0);

    return matrix;
}

template <class T>
std::vector<T> generate_ones_matrix(std::size_t height, std::size_t width) {
    auto num_elements = height * width;
    std::vector<T> matrix;
    matrix.reserve(num_elements);

    for (std::size_t i=0; i<num_elements; ++i)
        matrix.push_back(1);

    return matrix;
}

template <class T>
std::vector<T> generate_ladder_matrix(std::size_t height, std::size_t width, bool reverse=false) {
    auto num_elements = height * width;
    std::vector<T> matrix;
    matrix.reserve(num_elements);

    if (reverse) {
        for (std::size_t i=0; i<height; ++i)
            for (std::size_t j=0; j<width; ++j)
                matrix.push_back(j);
    } else {
        for (std::size_t i=0; i<height; ++i)
            for (std::size_t j=0; j<width; ++j)
                matrix.push_back(i);
    }

    return matrix;
}

template <class T>
std::vector<T> generate_random_sequence(std::size_t length, T min=0, T max=1, int seed=-1) {
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }

    std::vector<T> sequence;
    sequence.reserve(length);

    std::default_random_engine generator{static_cast<unsigned int>(seed)};
    if constexpr (std::is_same_v<T, int>) {
        std::uniform_int_distribution<int> dist(min, max);
        for (std::size_t i=0; i<length; ++i)
            sequence.push_back(dist(generator));

    } else {
        std::uniform_real_distribution<T> dist(min, max);
        for (std::size_t i=0; i<length; ++i)
            sequence.push_back(dist(generator));
    }

    return sequence;
}

template <class T>
std::vector<T> generate_constant_sequence(std::size_t length, T value) {
    return std::vector<T>(length, value);
}

template <class T>
std::vector<T> generate_batch_tensor(std::size_t batch, std::size_t height, std::size_t width, std::size_t depth, T min=0, T max=1, int seed=-1) {
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }

    auto num_elements = batch * height * width * depth;
    std::vector<T> matrix;
    matrix.reserve(num_elements);

    std::default_random_engine generator{static_cast<unsigned int>(seed)};
    if constexpr (std::is_same_v<T, int>) {
        std::uniform_int_distribution<int> dist(min, max);
        for (std::size_t i=0; i<num_elements; ++i)
            matrix.push_back(dist(generator));

    } else {
        std::uniform_real_distribution<T> dist(min, max);
        for (std::size_t i=0; i<num_elements; ++i)
            matrix.push_back(dist(generator));
    }

    return matrix;
}

template <class T>
std::vector<T> generate_constant_batch_tensor(std::size_t batch, std::size_t height, std::size_t width, std::size_t depth, T value) {
    auto num_elements = batch * height * width * depth;
    std::vector<T> matrix;
    matrix.reserve(num_elements);

    for (std::size_t i=0; i<num_elements; ++i)
        matrix.push_back(value);

    return matrix;
}

} // namespace tools

} // namespace combat