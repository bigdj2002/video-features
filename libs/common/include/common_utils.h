#pragma once

#include <vector>
#include <sstream>

extern void decompose_fps(double fps, int &numerator, int &denominator);

extern std::vector<int> parse_int_from_string(const std::string &input);