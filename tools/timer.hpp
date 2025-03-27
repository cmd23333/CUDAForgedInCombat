#pragma once
#include <chrono>
#include <iostream>

namespace combat {

namespace tools {

class Timer {
public:
    Timer(char const* tag) {
        _start_time = std::chrono::steady_clock::now();
        _tag = tag;

        std::cout << ">>>>>>>>>>>>> start " << _tag << " <<<<<<<<<<<<<" << std::endl;
    }

    ~Timer() {
        auto finish_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - _start_time).count();

        std::cout << "time cost: " << duration_ms << " milli-seconds" << std::endl;
        std::cout << ">>>>>>>>>>>>> finish " << _tag << " <<<<<<<<<<<<<" << std::endl;
    }

private:
    char const* _tag;
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
};

} // namespace tools

} // namespace combat