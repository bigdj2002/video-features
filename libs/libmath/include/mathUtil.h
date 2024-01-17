#pragma once

#include <cmath>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <complex>
#include <cassert>
#include <algorithm>

#include "xsimd.hpp"

namespace math_util
{
  // Bits
  bool IsPowerOfTwo(const uint32_t x) noexcept;
  int FloorLog2(const uint32_t x) noexcept;

  // Statistics
  template <typename T>
  double Accumulation(const std::vector<T> &v) noexcept
  {
    double sum = 0.0;
    for (auto x : v)
    {
      sum += x;
    }
    return sum;
  }

  template <typename T>
  float Accumulation_simd(const std::vector<T> &v) noexcept
  {
    float sum = 0.0;

    using batch_type = xsimd::batch<float>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = v.size() - v.size() % batch_size;

    for (size_t i = 0; i < algined_size; i += batch_size)
    {
      auto v_batch = xsimd::load_unaligned(&v[i]);
      sum += xsimd::reduce_add(v_batch);
    }

    for (size_t i = algined_size; i < v.size(); ++i)
    {
      sum += v[i];
    }
    return sum;
  }

  template <typename T>
  double Mean(const std::vector<T> &v) noexcept
  {
    return Accumulation(v) / v.size();
  }

  template <typename T>
  float Mean_simd(const std::vector<T> &v) noexcept
  {
    return Accumulation_simd(v) / v.size();
  }

  template <typename T>
  double Variance(
      const std::vector<T> &v,
      int ddof = 1) noexcept
  {
    assert(ddof >= 0);

    if ((int)v.size() < (1 + ddof))
      return 0.0f;

    double mean = Mean(v);
    double sum_sq_diff = 0.0f;

    for (auto x : v)
    {
      sum_sq_diff += (x - mean) * (x - mean);
    }

    double variance = sum_sq_diff / (v.size() - ddof);
    return variance;
  }

  template <typename T>
  float Variance_simd(
      const std::vector<T> &v,
      int ddof = 1) noexcept
  {
    assert(ddof >= 0);

    if ((int)v.size() < (1 + ddof))
      return 0.0f;

    float mean = Mean_simd(v);
    float sum_sq_diff = 0.0f;

    using batch_type = xsimd::batch<float>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = v.size() - v.size() % batch_size;

    for (size_t i = 0; i < algined_size; i += batch_size)
    {
      auto v_batch = xsimd::load_unaligned(&v[i]);
      sum_sq_diff += xsimd::reduce_add(xsimd::pow(xsimd::sub(v_batch, batch_type(mean)), 2));
    }

    for (size_t i = algined_size; i < v.size(); ++i)
    {
      sum_sq_diff += (v[i] - mean) * (v[i] - mean);
    }

    float variance = sum_sq_diff / (v.size() - ddof);
    return variance;
  }

  template <typename T>
  double StandardDeviation(
      const std::vector<T> &v,
      int ddof = 1) noexcept
  {
    return std::sqrt(Variance(v, ddof));
  }

  template <typename T>
  float StandardDeviation_simd(
      const std::vector<T> &v,
      int ddof = 1) noexcept
  {
    return std::sqrt(Variance_simd(v, ddof));
  }

  // reference: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
  template <typename T>
  double Kurtosis(const std::vector<T> &v)
  {
    const double s2 = Variance(v, 0);
    const double ex = Mean(v);
    double numer = 0.0;

    for (auto x : v)
    {
      numer += std::pow(x - ex, 4);
    }

    if (std::abs(s2) < std::numeric_limits<double>::epsilon())
      return 0.0;
    else
      return (numer / (v.size() * s2 * s2)) - 3.0;
  }

  template <typename T>
  float Kurtosis_simd(const std::vector<T> &v)
  {
    const float s2 = Variance_simd(v, 0);
    const float ex = Mean_simd(v);
    float numer = 0.0;

    using batch_type = xsimd::batch<float>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = v.size() - v.size() % batch_size;

    for (size_t i = 0; i < algined_size; i += batch_size)
    {
      auto v_batch = xsimd::load_unaligned(&v[i]);
      numer += xsimd::reduce_add(xsimd::pow(xsimd::sub(v_batch, batch_type(ex)), 4));
    }

    for (size_t i = algined_size; i < v.size(); ++i)
    {
      numer += std::pow(v[i] - ex, 4);
    }

    if (std::abs(s2) < std::numeric_limits<float>::epsilon())
      return 0.0;
    else
      return (numer / (v.size() * s2 * s2)) - 3.0;
  }

  // reference: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
  template <typename T>
  double Skewness(const std::vector<T> &v)
  {
    const double s = StandardDeviation(v, 0);
    const double ex = Mean(v);
    double numer = 0.0;

    for (auto x : v)
    {
      numer += std::pow(x - ex, 3);
    }

    if (std::abs(s) < std::numeric_limits<double>::epsilon())
      return 0.0;
    else
      return numer / (v.size() * std::pow(s, 3));
  }

  template <typename T>
  float Skewness_simd(const std::vector<T> &v)
  {
    const float s = StandardDeviation_simd(v, 0);
    const float ex = Mean_simd(v);
    float numer = 0.0;

    using batch_type = xsimd::batch<float>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = v.size() - v.size() % batch_size;

    for (size_t i = 0; i < algined_size; i += batch_size)
    {
      auto v_batch = xsimd::load_unaligned(&v[i]);
      numer += xsimd::reduce_add(xsimd::pow(xsimd::sub(v_batch, batch_type(ex)), 3));
    }

    for (size_t i = algined_size; i < v.size(); ++i)
    {
      numer += std::pow(v[i] - ex, 3);
    }

    if (std::abs(s) < std::numeric_limits<float>::epsilon())
      return 0.0;
    else
      return numer / (v.size() * std::pow(s, 3));
  }

  template <typename T>
  std::vector<uint32_t> Histogram(
      const std::vector<T> &v,
      uint32_t numBins)
  {
    assert(numBins > 0u);

    T minValue = *std::min_element(v.begin(), v.end());
    T maxValue = *std::max_element(v.begin(), v.end());

    // Calculate the bin width
    double binWidth = static_cast<double>(maxValue - minValue) / numBins;

    // Initialize the histogram bins with 0
    std::vector<uint32_t> histogram(numBins, 0u);

    if (std::abs(binWidth) < std::numeric_limits<double>::epsilon())
    {
      histogram[0] = v.size();
      return histogram;
    }

    // Iterate over the data and increment the appropriate bin
    for (auto value : v)
    {
      uint32_t binIndex = std::min(static_cast<uint32_t>(((value - minValue) / binWidth) + 0.5), numBins - 1u);
      assert(binIndex >= 0 && binIndex < numBins);
      histogram[binIndex] += 1.0;
    }

    return histogram;
  }

  template <typename T>
  double ShannonEntropy(const std::vector<T> &v)
  {
    uint32_t numBins = uint32_t(std::sqrt((double)v.size()) + 0.5);
    auto hist = Histogram(v, numBins);
    double entropy = 0.0;
    for (auto count : hist)
    {
      if (count > 0)
      {
        double prob = (double)count / v.size();
        entropy += (-prob * std::log2(prob));
      }
    }

    return entropy;
  }

  template <typename T>
  float ShannonEntropy_simd(const std::vector<T> &v)
  {
    uint32_t numBins = uint32_t(std::sqrt((float)v.size()) + 0.5);
    auto hist = Histogram(v, numBins);
    float entropy = 0.0;

    using batch_type = xsimd::batch<float>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = hist.size() - hist.size() % batch_size;

    for (size_t i = 0; i < algined_size; i += batch_size)
    {
      auto hist_batch = xsimd::batch_cast<float>(xsimd::load_unaligned(&hist[i]));
      auto prob_batch = xsimd::div(hist_batch, batch_type(v.size()));
      entropy += xsimd::reduce_add(xsimd::mul(xsimd::mul(batch_type(-1), prob_batch), xsimd::log2(prob_batch)));
    }

    for (size_t i = algined_size; i < hist.size(); ++i)
    {
      if (hist[i] > 0)
      {
        float prob = (float)hist[i] / v.size();
        entropy += (-prob * std::log2(prob));
      }
    }

    return entropy;
  }

  template <typename T>
  double InnerProduct(
      const std::vector<T> &x,
      const std::vector<T> &y)
  {
    if (x.size() != y.size())
      throw std::invalid_argument("x.size() must equal to y.size()");

    double sum = 0;
    for (size_t i = 0; i < x.size(); ++i)
      sum += double(x[i] * y[i]);

    return sum;
  }

  template <typename T>
  float InnerProduct_simd(
      const std::vector<T> &x,
      const std::vector<T> &y)
  {
    if (x.size() != y.size())
      throw std::invalid_argument("x.size() must equal to y.size()");

    using batch_type = xsimd::batch<float>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = x.size() - x.size() % batch_size;

    float sum = 0;
    for (size_t i = 0; i < algined_size; i += batch_size)
    {
      auto x_batch = xsimd::load_unaligned(&x[i]);
      auto y_batch = xsimd::load_unaligned(&y[i]);
      sum += xsimd::reduce_add(xsimd::mul(x_batch, y_batch));
    }

    for (size_t i = algined_size; i < x.size(); ++i)
    {
      sum += float(x[i] * y[i]);
    }

    return sum;
  }

  template <typename T>
  double PearsonCorrelationCoefficients(
      const std::vector<T> &x,
      const std::vector<T> &y)
  {
    if (x.size() != y.size() || x.empty())
      throw std::invalid_argument("x.size() must be equal to y.size()");

    double ex = Mean(x);
    double ey = Mean(y);

    double numer = 0;
    double denomx = 0;
    double denomy = 0;

    for (size_t i = 0; i < x.size(); ++i)
    {
      numer += (x[i] - ex) * (y[i] - ey);
      denomx += std::pow(x[i] - ex, 2);
      denomy += std::pow(y[i] - ey, 2);
    }

    double corr;
    double divisor = std::sqrt(denomx) * std::sqrt(denomy);

    if (std::abs(divisor) < std::numeric_limits<double>::epsilon())
      corr = 1.0;
    else
      corr = numer / divisor;

    return corr;
  }

  template <typename T>
  float PearsonCorrelationCoefficients_simd(
      const std::vector<T> &x,
      const std::vector<T> &y)
  {
    if (x.size() != y.size() || x.empty())
      throw std::invalid_argument("x.size() must be equal to y.size()");

    float ex = Mean_simd(x);
    float ey = Mean_simd(y);

    float numer = 0;
    float denomx = 0;
    float denomy = 0;

    using batch_type = xsimd::batch<T>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = x.size() - x.size() % batch_size;

    for (size_t i = 0; i < algined_size; i += batch_size)
    {
      auto x_batch = xsimd::load_unaligned(&x[i]);
      auto y_batch = xsimd::load_unaligned(&y[i]);
      auto x_diff_batch = xsimd::sub(x_batch, batch_type(ex));
      auto y_diff_batch = xsimd::sub(y_batch, batch_type(ey));

      numer += xsimd::reduce_add(xsimd::mul(x_diff_batch, y_diff_batch));
      denomx += xsimd::reduce_add(xsimd::pow(x_diff_batch, 2));
      denomy += xsimd::reduce_add(xsimd::pow(y_diff_batch, 2));
    }

    for (size_t i = algined_size; i < x.size(); ++i)
    {
      numer += (x[i] - ex) * (y[i] - ey);
      denomx += std::pow(x[i] - ex, 2);
      denomy += std::pow(y[i] - ey, 2);
    }

    float corr;
    float divisor = std::sqrt(denomx) * std::sqrt(denomy);

    if (std::abs(divisor) < std::numeric_limits<double>::epsilon())
      corr = 1.0;
    else
      corr = numer / divisor;

    return corr;
  }

  double Energy(const std::vector<std::complex<double>> &x) noexcept;

  template <typename T>
  void xsimd_memcpy(void *dest, const void *src, std::size_t count)
  {
    constexpr std::size_t batch_size = xsimd::batch<T>::size * sizeof(T);
    auto *fdest = static_cast<T *>(dest);
    const auto *fsrc = static_cast<const T *>(src);
    const size_t algined_size = count - count % batch_size;

    for (std::size_t i = 0; i < algined_size; i += batch_size)
    {
      xsimd::batch<T> src_batch = xsimd::load_unaligned(&fsrc[i]);
      xsimd::store_unaligned(&fdest[i], src_batch);      
    }

    std::size_t remaining = count % batch_size;
    if (remaining > 0)
    {
      std::memcpy(fdest + count - remaining, fsrc + count - remaining, remaining);
    }
  }
}

// template<typename T> std::complex<T> VDot(
//   const std::vector<std::complex<T>>& x,
//   const std::vector<std::complex<T>>& y)
// {
//   assert(x.size() == y.size() && !x.empty());

//   std::complex<T> output{};
//   for(size_t i=0; i<x.size(); ++i) {
//     output += std::conj(y[i]) * x[i];
//   }

//   return output;
// }

// template<typename T>
// std::complex<T> ComplexMean(const std::vector<std::complex<T>>& v)
// {
//   std::complex<T> sum{0, 0};

//   for(auto k: v) {
//     sum += k;
//   }

//   return std::complex<T>{
//     sum.real() / v.size(),
//     sum.imag() / v.size()
//   };
// }
