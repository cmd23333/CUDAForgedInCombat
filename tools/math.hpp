#pragma once
#include <iostream>
#include <vector>
#include <cmath>

namespace combat {

namespace tools {

int least_power_of_two_leq_to_number(int number) {
    /*
        返回大于等于 number 的第一个 2 的正整数幂
    */
    int target = 1;
    for (; target < number; target *= 2);
    return target;
}

} // namespace tools

} // namespace combat