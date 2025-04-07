#pragma once

#include <string.h>
#include <cmath>
#include <numeric>

#include "tools/show.hpp"
#include "combat/matrix_multiply/host_matrix_multiply.hpp"

namespace combat {

namespace flash_attention {

template<class T>
std::vector<T> scaled_matmul(T const *a, T const *b, std::size_t m, std::size_t k, std::size_t n) {
    /*
        a: [m, k], b: [n, k]
        calculate a*b.T: [m, n]
    */
    auto out = std::vector<T>(m*n, 0);

    for (std::size_t i=0; i<m; ++i) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t k_=0; k_<k; ++k_) {
                out[i*n+j] += a[i*k+k_] * b[j*k+k_];
            }
            out[i*n+j] /= std::sqrt(k);
        }
    }

    return out;
}

template<class T>
void online_softmax(
    T *o_tile, T const *q_tile, T const *k_tile, T const *v_tile,
    T *old_max, T *old_sum,
    std::size_t q_tile_height, std::size_t depth, std::size_t k_tile_height
) {
    /*
        逻辑上 q_tile * k_tile.T, 实际上我们不会对 k_tile 做转置
        (因此没复用 matrix_multiply::matrix_multiply_host, TODO(cmd2333): 给 matrix_multiply_host 加入相关功能)
        o_tile = o_tile * exp(old_max-new_max) * old_sum / new_sum + exp(q_tile * k_tile.T - new_max) / new_sum * v_tile
    */
    tools::show_matrix(old_max, 1, q_tile_height, "old max");
    tools::show_matrix(old_sum, 1, q_tile_height, "old sum");

    auto a_tile_vec = scaled_matmul(q_tile, k_tile, q_tile_height, depth, k_tile_height);
    T *a_tile = a_tile_vec.data();
    tools::show_matrix(a_tile, q_tile_height, k_tile_height, "q@k.T / sqrt(d)");

    T *new_max = (T *)malloc(q_tile_height*sizeof(T));
    T *new_sum = (T *)malloc(q_tile_height*sizeof(T));
    memcpy(new_max, old_max, q_tile_height*sizeof(T));

    const auto a_tile_height = q_tile_height;
    const auto a_tile_width = k_tile_height;

    for (std::size_t a_tile_row_index=0; a_tile_row_index < a_tile_height; ++a_tile_row_index) {
        for (std::size_t a_tile_col_index=0; a_tile_col_index < a_tile_width; ++a_tile_col_index) {
            new_max[a_tile_row_index] = std::max(new_max[a_tile_row_index], a_tile[a_tile_row_index * a_tile_width + a_tile_col_index]);
        }
    }

    for (std::size_t a_tile_row_index=0; a_tile_row_index < a_tile_height; ++a_tile_row_index) {
        new_sum[a_tile_row_index] = 0;
        T const sum_adjust_coef = std::exp(old_max[a_tile_row_index] - new_max[a_tile_row_index]);
        new_sum[a_tile_row_index] = sum_adjust_coef * old_sum[a_tile_row_index];
        for (std::size_t a_tile_col_index=0; a_tile_col_index < a_tile_width; ++a_tile_col_index) {
            new_sum[a_tile_row_index] += std::exp(a_tile[a_tile_row_index * a_tile_width + a_tile_col_index] - new_max[a_tile_row_index]);
        }
    }

    tools::show_matrix(new_max, 1, q_tile_height, "max");
    tools::show_matrix(new_sum, 1, q_tile_height, "sum");

    const auto o_tile_height = q_tile_height;
    const auto o_tile_width = depth;
    const auto v_tile_height = k_tile_height;
    const auto v_tile_width = depth;
    for (std::size_t o_tile_row_index=0; o_tile_row_index < o_tile_height; ++o_tile_row_index) {
        // local sum: [q_tile_height, 1]
        auto const o_adjust_coef = std::exp(old_max[o_tile_row_index] - new_max[o_tile_row_index]) * old_sum[o_tile_row_index] / new_sum[o_tile_row_index];
        for (std::size_t o_tile_col_index=0; o_tile_col_index < o_tile_width; ++o_tile_col_index) {
            o_tile[o_tile_row_index * depth + o_tile_col_index] = o_tile[o_tile_row_index * depth + o_tile_col_index] * o_adjust_coef;

            T o_ij = 0;
            // a_tile_width == k_tile_height == v_tile_height
            // a_tile: [m, n], v_tile: [n, d]
            // o_tile: [m, d]
            for (std::size_t a_tile_col_index=0; a_tile_col_index < a_tile_width; ++a_tile_col_index) {
                T const a_tile_ik = std::exp(a_tile[o_tile_row_index * a_tile_width + a_tile_col_index] - new_max[o_tile_row_index]) / new_sum[o_tile_row_index];
                o_ij += a_tile_ik * v_tile[a_tile_col_index * v_tile_width + o_tile_col_index];
            }

            o_tile[o_tile_row_index * depth + o_tile_col_index] += o_ij;
        }
    }

    memcpy(old_max, new_max, q_tile_height*sizeof(T));
    memcpy(old_sum, new_sum, q_tile_height*sizeof(T));
    free(new_max);
    free(new_sum);
}


template<class T>
void online_single_head_attention(
    T *single_O, T const *single_Q, T const *single_K, T const *single_V,
    std::size_t seq_len, std::size_t depth
) {
    /*
        single batch, single head.
        single_O, single_Q, single_K, single_V.shape = [seq_len, depth]
    */

    size_t const tile_width = depth;
    // 这些个值要根据硬件来适配
    // q_tile: [m, d], k_tile: [n, d], att_tile: [m, n], v_tile: [n, d], o_tile: [m, d]
    // 因此 k_tile 和 v_tile 的高度要一致, q_tile 和 o_tile 的高度要一致
    size_t const q_tile_height = 4;
    auto const k_tile_height = 4;
    auto const v_tile_height = k_tile_height;
    auto const o_tile_height = q_tile_height;

    // load q, 这里假定了 seq_len 可以被 tile_height 整除
    size_t const num_q_tile = seq_len / q_tile_height;
    size_t const num_k_tile = seq_len / k_tile_height;
    size_t const num_v_tile = seq_len / v_tile_height;
    size_t const num_o_tile = seq_len / o_tile_height;

    // max_ 和 sum_ 保存了 attention 矩阵每一行当前的 最大值/去指数后求和
    T *max_ = (T *)malloc(seq_len*sizeof(T));
    for (std::size_t i=0; i<seq_len; ++i) max_[i] = std::numeric_limits<T>::lowest();
    T *sum_ = (T *)malloc(seq_len*sizeof(T));
    memset(sum_, 0, seq_len*sizeof(T));

    // 我们保证 num_k_tile == num_v_tile
    for (size_t k_or_v_tile_index=0; k_or_v_tile_index<num_k_tile; ++k_or_v_tile_index) {
        // 外层循环, 取出 k_tile 和 v_tile; tile_width == depth
        auto const shift_to_k_or_v_tile = k_or_v_tile_index * v_tile_height * tile_width;
        T const *k_tile = single_K + shift_to_k_or_v_tile;
        T const *v_tile = single_V + shift_to_k_or_v_tile;

        // 保证 num_q_tile == num_v_tile
        for (size_t q_or_v_tile_index=0; q_or_v_tile_index < num_q_tile; ++q_or_v_tile_index) {
            // 内层循环, 取出 q_tile 和 o_tile
            auto const shift_to_q_or_o_tile = q_or_v_tile_index * q_tile_height * tile_width;
            T const *q_tile = single_Q + q_or_v_tile_index * tile_width * q_tile_height;
            T *o_tile = single_O + q_or_v_tile_index * tile_width * q_tile_height;

            online_softmax(
                o_tile, q_tile, k_tile, v_tile,
                &max_[q_or_v_tile_index * q_tile_height], &sum_[q_or_v_tile_index * q_tile_height],
                q_tile_height, depth, k_tile_height
            );
        }
    }

    free(max_);
    free(sum_);
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
            T const *single_Q = Q + shift_to_current_2d_matrix; // [seq_len, depth]
            T const *single_K = K + shift_to_current_2d_matrix;
            T const *single_V = V + shift_to_current_2d_matrix;
            T *single_O = out + shift_to_current_2d_matrix;

            online_single_head_attention(single_O, single_Q, single_K, single_V, seq_len, depth);
        }
    }
}

} // namespace flash_attention

} // namespace combat