add_executable(flash_attention_combat ${CMAKE_CURRENT_SOURCE_DIR}/examination.cu)
target_compile_options(flash_attention_combat PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)