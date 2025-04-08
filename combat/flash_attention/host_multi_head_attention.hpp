#pragma once
#include <cmath>
#include <numeric>
#include "combat/matrix_multiply/host_matrix_multiply.hpp"
#include "tools/show.hpp"

namespace combat {

namespace flash_attention {

template<class T>
void single_head_attention(
    T *o, T const *q, T const *k, T const *v, 
    std::size_t seq_len, std::size_t depth
) {
    /*
        all of them have shape = [seq_len, depth]
    */

    const auto num_elements = seq_len * seq_len;
    T *attention = (T *)malloc(sizeof(T) * num_elements);
    
    for (std::size_t i=0; i<seq_len; ++i) {
        for (std::size_t j=0; j<seq_len; ++j) {
            T out_ij = 0;
            for (std::size_t k_=0; k_<depth; ++k_) {
                out_ij += q[i * depth + k_] * k[j * depth + k_];
            }
            attention[i * seq_len + j] = out_ij / std::sqrt(depth);
        }
    }

    tools::show_matrix(attention, seq_len, seq_len, "q@k/sqrt(d)");

    for (std::size_t i=0; i<seq_len; ++i) {
        T max_row_value = std::numeric_limits<T>::min();
        for (std::size_t j=0; j<seq_len; ++j) {
            if (attention[i*seq_len+j] > max_row_value)
                max_row_value = attention[i*seq_len+j];
        }
        T row_sum = 0;
        for (std::size_t j=0; j<seq_len; ++j) {
            attention[i*seq_len+j] = std::exp(attention[i*seq_len+j] - max_row_value);
            row_sum += attention[i*seq_len+j];
        }
        for (std::size_t j=0; j<seq_len; ++j) {
            attention[i*seq_len+j] /= row_sum;
        }
    }

    for (std::size_t i=0; i<seq_len; ++i) {
        for (std::size_t k_=0; k_<seq_len; ++k_) {
            T const t = attention[i * seq_len + k_];
            for (std::size_t j=0; j<depth; ++j) {
                o[i * depth + j] += t * v[k_ * depth + j];
            }
        }
    }

    tools::show_matrix(attention, seq_len, seq_len, "attention");
    free(attention);
}

template<class T>
void multi_head_attention_host(
    T *out, T const *Q, T const *K, T const *V, 
    std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t depth
) {
    /*
        提示: Q,K,V.shape = [batch, num_head, seq_len, depth]
    */
    for (size_t batch_index=0; batch_index < batch_size; ++batch_index) {
        for (size_t head_index=0; head_index < num_heads; ++head_index) {
            auto const shift_to_current_2d_matrix = batch_index * (num_heads * seq_len * depth) + head_index * (seq_len * depth);
            T const *single_Q = Q + shift_to_current_2d_matrix; // [seq_len, depth]
            T const *single_K = K + shift_to_current_2d_matrix;
            T const *single_V = V + shift_to_current_2d_matrix;
            T *single_O = out + shift_to_current_2d_matrix;

            single_head_attention(single_O, single_Q, single_K, single_V, seq_len, depth);
        }
    }
}

} // namespace flash_attention

} // namespace combat