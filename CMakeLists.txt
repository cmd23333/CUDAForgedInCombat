cmake_minimum_required(VERSION 3.20)

project(CUDA_FORGED_IN_COMBAT)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(BUILD_TESTING "Option for building unit test" ON)
if(BUILD_TESTING)
    add_executable(ut_timer ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/test_timer.cpp)
    add_executable(ut_generator ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/test_generator.cpp)
endif(BUILD_TESTING)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/combat/reduce_sum)