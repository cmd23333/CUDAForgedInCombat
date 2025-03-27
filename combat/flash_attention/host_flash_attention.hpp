#pragma once

namespace combat {

namespace flash_attention {

template<class T>
void flash_attention_host(
    T *out, T const *Q, T const *K, T const *V, 
    std::size_t batch, std::size_t height, std::size_t width, std::size_t depth
) {
    /*
        提示: Q,K,V.shape = [batch, num_head, seq_len, depth]
    */
}

} // namespace flash_attention

} // namespace combat