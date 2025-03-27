#pragma once
#include <iostream>
#include <vector>

namespace combat {

namespace tools {

template <class T>
void show_matrix(T const *matrix, std::size_t height, std::size_t width, char const *tag=nullptr) {
    if (tag != nullptr)
        std::cout << "show matrix: " << tag << std::endl;

    for (std::size_t i=0; i<height; ++i) {
        auto base = i*width;
        for (std::size_t j=0; j<width; ++j) {
            std::cout << matrix[base+j] << " ";
        }
        std::cout << std::endl;
    }
}

template <class T>
void show_matrix(std::vector<T> const &matrix, std::size_t width, char const *tag=nullptr) {
    std::size_t height = matrix.size() / width;
    show_matrix<T>(matrix.data(), height, width, tag);
}

template <class T>
void show_matrix_first_row(std::vector<T> const &matrix, std::size_t width, char const *tag=nullptr) {
    if (tag != nullptr)
        std::cout << "show matrix first row: " << tag << std::endl;

    std::size_t height = 1;
    show_matrix<T>(matrix.data(), height, width);
}

template <class T>
void show_matrix_first_column(T const *matrix, std::size_t height, std::size_t width, char const *tag=nullptr) {
    if (tag != nullptr)
        std::cout << "show matrix first column: " << tag << std::endl;

    for (std::size_t i=0; i<height; ++i) {
        auto base = i*width;
        std::cout << matrix[base] << std::endl;
    }
}

template <class T>
void show_matrix_first_column(std::vector<T> const &matrix, std::size_t width, char const *tag=nullptr) {
    std::size_t height = matrix.size() / width;
    show_matrix_first_column(matrix.data(), height, width, tag);
}

template <class T>
void show_sequence(std::vector<T> const &sequence, std::size_t number_shown=0, char const *tag=nullptr) {
    if (number_shown == 0)
        number_shown = sequence.size();

    if (tag != nullptr)
        std::cout << "show sequence: " << tag << ", (first " << number_shown << " number will be shown)" << std::endl;
    else
        std::cout << "show sequence (first " << number_shown << " number will be shown)" << std::endl;

    for (std::size_t i=0; i<number_shown; ++i)
        std::cout << sequence[i] << " ";

    std::cout << std::endl;
}

} // namespace tools

} // namespace combat