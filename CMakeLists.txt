cmake_minimum_required(VERSION 3.20)

# 奇怪的编译指令, 可能是我运行在 Win11 的 WSL Ubuntu 虚拟机上
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(CUDA_FORGED_IN_COMBAT LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -G -keep") # 为了使用 Nsight + cuda-gdb 调试

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 89)  # 我用的 4070

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(BUILD_TESTING "Option for building unit test" ON)
if(BUILD_TESTING)
    add_executable(ut_timer ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/test_timer.cpp)
    add_executable(ut_generator ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/test_generator.cpp)
endif(BUILD_TESTING)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/combat/reduce_sum)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/combat/matrix_multiply)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/combat/flash_attention)