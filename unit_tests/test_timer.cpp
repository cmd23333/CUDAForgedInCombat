#include "tools/timer.hpp"

#include <thread>
#include <chrono>

using namespace combat::tools;

int main() {
    {
        auto t = Timer("unit test");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}
