#include "common_utils.h"

static int gcd(int a, int b)
{
  if (b == 0)
    return a;
  return gcd(b, a % b);
}

void decompose_fps(double fps, int &numerator, int &denominator)
{
  denominator = 1;

  while (fps != (int)fps)
  {
    fps *= 10;
    denominator *= 10;
  }

  numerator = (int)fps;

  int commonDivisor = gcd(numerator, denominator);
  numerator /= commonDivisor;
  denominator /= commonDivisor;
}

std::vector<int> parse_int_from_string(const std::string &input)
{
  std::vector<int> result;
  std::stringstream ss(input);
  std::string item;

  while (getline(ss, item, ':'))
  {
    try
    {
      result.push_back(std::stoi(item));
    }
    catch (const std::invalid_argument &e)
    {
      throw std::invalid_argument("List items must be integers");
    }
  }
  return result;
}