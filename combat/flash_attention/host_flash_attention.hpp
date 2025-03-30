#pragma once
#include <cmath>
#include <numeric>
#include "combat/matrix_multiply/host_matrix_multiply.hpp"

namespace combat {

namespace flash_attention {

template<class T>
void online_softmax(
    T *o_tile, T const *q_tile, T const *k_tile, T const *v_tile,
    T *old_max, T *old_sum,
    std::size_t q_tile_height, std::size_t k_tile_height, std::size_t depth
) {
    /*
        o_tile = o_tile * exp(old_max-new_max) * old_sum / new_sum + exp(q_tile * k_tile.T - new_max) / new_sum * v_tile
    */
    T a_tile[q_tile_height][k_tile_height];
    matrix_multiply::matrix_multiply_host(&(a_tile[0][0]), q_tile, k_tile, q_tile_height, depth, k_tile_height);

    T new_max[q_tile_height];
    for (std::size_t a_row_index=0; a_row_index < q_tile_height; ++a_row_index) {
        new_max[a_row_index] = std::numeric_limits<T>::min;
        for (std::size_t a_col_index=0; a_col_index < k_tile_height; ++a_col_index) {
            new_max[a_row_index] = std::max(new_max[a_row_index], a_tile[a_row_index][a_col_index]);
        }
    }

    T new_sum[q_tile_height];
    for (std::size_t a_row_index=0; a_row_index < q_tile_height; ++a_row_index) {
        new_sum[a_row_index] = 0;
        T const sum_adjust_coef = std::exp(old_max[a_row_index] - new_max[a_row_index]);
        new_sum[a_row_index] = sum_adjust_coef * old_sum[a_row_index]
        for (std::size_t a_col_index=0; a_col_index < k_tile_height; ++a_col_index) {
            new_sum[a_row_index] += std::exp(a_tile[a_row_index][a_col_index] - new_max[a_row_index]);
        }
    }

    for (std::size_t a_row_index=0; a_row_index < q_tile_height; ++a_row_index) {
        auto const o_adjust_coef = std::exp(old_max[a_row_index] - new_max[a_row_index]) * old_sum[a_row_index] / new_sum[a_row_index];
        for (std::size_t a_col_index=0; a_col_index < k_tile_height; ++a_col_index) {
            o_tile[a_row_index][a_col_index] = o_tile[a_row_index][a_col_index] * o_adjust_coef;
            o_tile[a_row_index][a_col_index] += a_tile[a_row_index][a_col_index] * v_tile[a_col_index][a_row_index];
        }
    }
    
    old_max = new_max;
    old_sum = new_sum;
}

template<class T>
void flash_attention_host(
    T *out, T const *Q, T const *K, T const *V, 
    std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t depth
) {
    /*
        提示: Q,K,V.shape = [batch, num_head, seq_len, depth]
    */
    for (size_t batch_index=0; batch_index < batch_size; ++batch_index) {
        for (size_t head_index=0; head_index < num_heads; ++head_index) {
            auto const shift_to_current_2d_matrix = batch_index * (num_heads * seq_len * depth) + head_index * (seq_len * depth);
            int const *single_Q = Q + shift_to_current_2d_matrix; // [seq_len, depth]
            int const *single_K = K + shift_to_current_2d_matrix;
            int const *single_V = V + shift_to_current_2d_matrix;
            int *single_O = out + shift_to_current_2d_matrix;

            size_t const tile_width = depth;
            // 这些个值要根据硬件来适配
            size_t const q_tile_height = 16;
            auto const k_tile_height = q_tile_height;
            auto const v_tile_height = q_tile_height;

            // load q, 这里假定了 seq_len 可以被 tile_height 整除
            size_t const num_q_tile = seq_len / q_tile_height;
            size_t const num_k_tile = seq_len / k_tile_height;
            size_t const num_v_tile = seq_len / v_tile_height;
            
            T *max_[seq_len] = {};
            T *sum_[seq_len] = {};

            // 我们保证 num_k_tile == num_v_tile
            for (size_t k_or_v_tile_index=0; k_or_v_tile_index<num_k_tile; ++k_or_v_tile_index) {
                auto const shift_to_k_or_v_tile = k_or_v_tile_index * v_tile_height * tile_width;
                int const *k_tile = single_K + shift_to_k_or_v_tile;
                int const *v_tile = single_V + shift_to_k_or_v_tile;

                for (size_t q_tile_index=0; q_tile_index < num_q_tile; ++q_tile_index) {
                    // [q_tile_height, tile_width] = [16, depth]
                    int const *q_tile = single_Q + q_tile_index * tile_width * q_tile_height; 
                    // [q_tile_height, tile_width] = [16, depth]
                    int *o_tile = single_O + q_tile_index * tile_width * q_tile_height;
                    online_softmax(
                        o_tile, q_tile, k_tile, v_tile, 
                        &max_[num_q_tile*q_tile_height], &sum_[num_q_tile*q_tile_height], 
                        q_tile_height, k_tile_height, depth
                    );
                }
            }
        }
    }
}

} // namespace flash_attention

} // namespace combat