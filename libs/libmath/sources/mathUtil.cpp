#include "mathUtil.h"

template <typename T>
double CalcEnergyComplex(const std::vector<std::complex<T>> &x) noexcept
{
  double output = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
  {
    output += std::abs(std::conj(x[i]) * x[i]);
  }

  return output;
}

template <typename T>
double CalcEnergy(const std::vector<T> &x) noexcept
{
  double output = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
  {
    output += std::abs(x[i] * x[i]);
  }

  return output;
}

namespace math_util
{
  bool IsPowerOfTwo(const uint32_t x) noexcept
  {
    return (x > 0) && ((x & (x - 1)) == 0);
  }

  int FloorLog2(const uint32_t x) noexcept
  {
    if (x == 0)
    {
      return -1;
    }

    return 31 - __builtin_clz(x);
  }

  double Energy(const std::vector<std::complex<double>> &x) noexcept
  {
    return CalcEnergyComplex(x);
  }
}