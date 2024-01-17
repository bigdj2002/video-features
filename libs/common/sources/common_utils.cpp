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