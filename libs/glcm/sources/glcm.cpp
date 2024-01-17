#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <iostream>

#include "glcm.h"
#include "mathUtil.h"
#include "xsimd.hpp"

namespace glcm
{
  std::vector<double> graycomatrix(
      const uint8_t *data,
      int stride,
      int depth,
      int width,
      int height,
      int distance,
      double angle,
      int log2Levels,
      bool normalize)
  {
    if (log2Levels < 0)
      throw std::invalid_argument("log2Levels must be a positive value");

    if (depth > 16 || depth < 1)
      throw std::invalid_argument("depth must be in range 1~16");

    if (log2Levels > depth)
      throw std::invalid_argument("log2Levels must be less than or equal to depth");

    if (data == nullptr)
      throw std::invalid_argument("data is null");

    const size_t levels = 1 << log2Levels;
    const size_t matrix_size = 1 << (log2Levels * 2u);
    int dx = distance * std::round(std::cos(angle));
    int dy = distance * std::round(std::sin(angle));
    uint64_t total = 0u;

    std::vector<uint32_t> hist(matrix_size);
    std::vector<double> comatrix(matrix_size, 0.0f);
    std::vector<int> convert(levels);
    const int *LUT = nullptr;

    if (log2Levels != depth)
    {
      const int rshift = depth - log2Levels;
      const int add = 1 << (rshift - 1);

      for (int i = 0; i < (1 << depth); i++)
      {
        convert[i] = (i + add) >> rshift;
      }
      LUT = convert.data();
    }

    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        int x = j + dx;
        int y = i + dy;

        if (x >= 0 && x < width && y >= 0 && y < height)
        {
          int r;
          int c;

          if (depth <= 8)
          {
            r = data[i * stride + j];
            c = data[y * stride + x];
          }
          else
          {
            r = *((uint16_t *)&data[i * stride + j * 2u]);
            c = *((uint16_t *)&data[y * stride + x * 2u]);
          }

          if (LUT)
          {
            r = LUT[r];
            c = LUT[c];
          }

          hist[r * levels + c]++;
          total++;
        }
      }
    }

    if (total > 0)
    {
      if (normalize)
      {
        double mult = 1.0 / total;
        for (size_t i = 0; i < matrix_size; i++)
          comatrix[i] = hist[i] * mult;
      }
      else
      {
        for (size_t i = 0; i < matrix_size; i++)
          comatrix[i] = double(hist[i]);
      }
    }

    return comatrix;
  }

  std::vector<float> graycomatrix_simd(
      const uint8_t *data,
      int stride,
      int depth,
      int width,
      int height,
      int distance,
      double angle,
      int log2Levels,
      bool normalize)
  {
    if (log2Levels < 0)
      throw std::invalid_argument("log2Levels must be a positive value");

    if (depth > 16 || depth < 1)
      throw std::invalid_argument("depth must be in range 1~16");

    if (log2Levels > depth)
      throw std::invalid_argument("log2Levels must be less than or equal to depth");

    if (data == nullptr)
      throw std::invalid_argument("data is null");

    const size_t levels = 1 << log2Levels;
    const size_t matrix_size = 1 << (log2Levels * 2u);
    int dx = distance * std::round(std::cos(angle));
    int dy = distance * std::round(std::sin(angle));
    uint64_t total = 0u;

    std::vector<uint32_t> hist(matrix_size);
    std::vector<float> comatrix(matrix_size, 0.0f);
    std::vector<int> convert(levels);
    const int *LUT = nullptr;

    if (log2Levels != depth)
    {
      const int rshift = depth - log2Levels;
      const int add = 1 << (rshift - 1);

      using convert_type = xsimd::batch<int32_t>;
      std::size_t batch_size = convert_type::size;
      const size_t aligned_size = (1 << depth) - (1 << depth) % batch_size;
      convert_type base_i_batch(0, 1, 2, 3, 4, 5, 6, 7);

      for (size_t i = 0, cnt = 0; i < aligned_size; i += batch_size, ++cnt)
      {
        auto i_batch = xsimd::add(base_i_batch, convert_type(cnt * batch_size));
        convert_type convert_batch = xsimd::bitwise_rshift(xsimd::add(i_batch, convert_type(add)), convert_type(rshift));
        for (std::size_t j = 0; j < batch_size; ++j)
          convert.push_back(convert_batch.get(j));
      }
      LUT = convert.data();
    }

    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        int x = j + dx;
        int y = i + dy;

        if (x >= 0 && x < width && y >= 0 && y < height)
        {
          int r;
          int c;

          if (depth <= 8)
          {
            r = data[i * stride + j];
            c = data[y * stride + x];
          }
          else
          {
            r = *((uint16_t *)&data[i * stride + j * 2u]);
            c = *((uint16_t *)&data[y * stride + x * 2u]);
          }

          if (LUT)
          {
            r = LUT[r];
            c = LUT[c];
          }

          hist[r * levels + c]++;
          total++;
        }
      }
    }

    if (total > 0)
    {
      if (normalize)
      {
        float mult = 1.0 / total;
        for (size_t i = 0; i < matrix_size; i++)
          comatrix[i] = hist[i] * mult;
      }
      else
      {
        for (size_t i = 0; i < matrix_size; i++)
          comatrix[i] = float(hist[i]);
      }
    }

    return comatrix;
  }

  double graycoprops(
      const std::vector<double> &comatrix,
      property prop)
  {
    const size_t dim = comatrix.size();
    const double eps = std::numeric_limits<double>::epsilon();

    if (!math_util::IsPowerOfTwo(dim))
      throw std::invalid_argument("Comatrix size must be power of two integer number.");

    const size_t log2dim = math_util::FloorLog2(dim);
    const size_t half_log2dim = log2dim / 2;

    if (log2dim != 2u * half_log2dim)
      throw std::invalid_argument("Unsupported comatrix size. size may be too small");

    const size_t mask = (1 << half_log2dim) - 1u;
    double result = 0;

    switch (prop)
    {
    case dissimilarity:
      for (size_t i = 0; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;
        result += comatrix[i] * std::abs(x - y);
      }
      break;

    case contrast:
      for (size_t i = 0; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;
        result += comatrix[i] * ((x - y) * (x - y));
      }
      break;

    case homogeneity:
      for (size_t i = 0; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;
        result += comatrix[i] / (1 + ((x - y) * (x - y)));
      }
      break;

    case energy:
      for (size_t i = 0; i < dim; i++)
      {
        result += comatrix[i] * comatrix[i];
      }
      result = std::sqrt(result);
      break;

    case correlation:
    {
      const size_t levels = 1u << half_log2dim;
      std::vector<double> row_sum(levels, 0.0);
      std::vector<double> col_sum(levels, 0.0);
      double mu_i = 0.0;
      double mu_j = 0.0;
      double cov = 0.0;
      double var_i = 0.0;
      double var_j = 0.0;

      for (size_t i = 0; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;

        row_sum[y] += comatrix[i];
        col_sum[x] += comatrix[i];
      }

      for (size_t i = 0; i < levels; i++)
      {
        mu_i += i * row_sum[i];
        mu_j += i * col_sum[i];
      }

      for (size_t i = 0; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;
        cov += comatrix[i] * ((y - mu_i) * (x - mu_j));
      }

      for (size_t i = 0; i < levels; i++)
      {
        var_i += (i - mu_i) * (i - mu_i) * row_sum[i];
        var_j += (i - mu_j) * (i - mu_j) * col_sum[i];
      }

      if (abs(var_i) < eps || abs(var_j) < eps)
        result = 1.0;
      else
        result = cov / sqrt(var_i * var_j);
    }
    break;

    case entropy:
      for (size_t i = 0; i < dim; i++)
      {
        if (std::abs(comatrix[i]) > eps)
          result += -comatrix[i] * std::log2(comatrix[i]);
      }
      break;
    default:
      break;
    }

    return result;
  }

  double graycoprops_simd(
      const std::vector<float> &comatrix,
      property prop)
  {
    const size_t dim = comatrix.size();
    const double eps = std::numeric_limits<double>::epsilon();

    if (!math_util::IsPowerOfTwo(dim))
      throw std::invalid_argument("Comatrix size must be power of two integer number.");

    const size_t log2dim = math_util::FloorLog2(dim);
    const size_t half_log2dim = log2dim / 2;

    if (log2dim != 2u * half_log2dim)
      throw std::invalid_argument("Unsupported comatrix size. size may be too small");

    const size_t mask = (1 << half_log2dim) - 1u;
    double result = 0;

    using batch_type = xsimd::batch<int32_t>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = dim - dim % batch_size;
    batch_type base_i_batch(0, 1, 2, 3, 4, 5, 6, 7);

    switch (prop)
    {
    case dissimilarity:
    {
      for (size_t i = 0, cnt = 0; i < algined_size; i += batch_size, ++cnt)
      {
        auto i_batch = xsimd::add(base_i_batch, batch_type(cnt * batch_size));
        auto x_batch = xsimd::bitwise_and(i_batch, batch_type(mask));
        auto y_batch = xsimd::bitwise_rshift(i_batch, batch_type(half_log2dim));
        auto result_batch = xsimd::mul(xsimd::load_unaligned(&comatrix[i]), xsimd::batch_cast<float>(xsimd::abs(xsimd::sub(x_batch, y_batch))));
        result += xsimd::reduce_add(result_batch);
      }

      for (size_t i = algined_size; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;
        result += comatrix[i] * std::abs(x - y);
      }
    }
    break;

    case contrast:
    {
      for (size_t i = 0, cnt = 0; i < algined_size; i += batch_size, ++cnt)
      {
        auto i_batch = xsimd::add(base_i_batch, batch_type(cnt * batch_size));
        auto x_batch = xsimd::bitwise_and(i_batch, batch_type(mask));
        auto y_batch = xsimd::bitwise_rshift(i_batch, batch_type(half_log2dim));
        auto result_batch = xsimd::mul(xsimd::load_unaligned(&comatrix[i]), xsimd::batch_cast<float>(xsimd::pow(xsimd::sub(x_batch, y_batch), 2)));
        result += xsimd::reduce_add(result_batch);
      }

      for (size_t i = algined_size; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;
        result += comatrix[i] * ((x - y) * (x - y));
      }
    }
    break;

    case homogeneity:
    {
      for (size_t i = 0, cnt = 0; i < algined_size; i += batch_size, ++cnt)
      {
        auto i_batch = xsimd::add(base_i_batch, batch_type(cnt * batch_size));
        auto x_batch = xsimd::bitwise_and(i_batch, batch_type(mask));
        auto y_batch = xsimd::bitwise_rshift(i_batch, batch_type(half_log2dim));
        auto result_batch = xsimd::div(xsimd::load_unaligned(&comatrix[i]), xsimd::add(xsimd::batch<float>(1), xsimd::batch_cast<float>(xsimd::pow(xsimd::sub(x_batch, y_batch), 2))));
        result += xsimd::reduce_add(result_batch);
      }

      for (size_t i = algined_size; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;
        result += comatrix[i] / (1 + ((x - y) * (x - y)));
      }
    }

    break;

    case energy:
    {
      for (size_t i = 0, cnt = 0; i < algined_size; i += batch_size, ++cnt)
      {
        auto comat_batch = xsimd::load_unaligned(&comatrix[i]);
        auto result_batch = xsimd::pow(comat_batch, 2);
        result += xsimd::reduce_add(result_batch);
      }

      for (size_t i = algined_size; i < dim; i++)
      {
        result += comatrix[i] * comatrix[i];
      }
      result = std::sqrt(result);
    }

    break;

    case correlation:
    {
      const size_t levels = 1u << half_log2dim;
      std::vector<double> row_sum(levels, 0.0);
      std::vector<double> col_sum(levels, 0.0);
      double mu_i = 0.0;
      double mu_j = 0.0;
      double cov = 0.0;
      double var_i = 0.0;
      double var_j = 0.0;

      for (size_t i = 0; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;

        row_sum[y] += comatrix[i];
        col_sum[x] += comatrix[i];
      }

      for (size_t i = 0; i < levels; i++)
      {
        mu_i += i * row_sum[i];
        mu_j += i * col_sum[i];
      }

      for (size_t i = 0; i < dim; i++)
      {
        int x = i & mask;
        int y = i >> half_log2dim;
        cov += comatrix[i] * ((y - mu_i) * (x - mu_j));
      }

      for (size_t i = 0; i < levels; i++)
      {
        var_i += (i - mu_i) * (i - mu_i) * row_sum[i];
        var_j += (i - mu_j) * (i - mu_j) * col_sum[i];
      }

      if (abs(var_i) < eps || abs(var_j) < eps)
        result = 1.0;
      else
        result = cov / sqrt(var_i * var_j);
    }
    break;

    case entropy:
    {
      for (size_t i = 0, cnt = 0; i < algined_size; i += batch_size, ++cnt)
      {
        auto comat_batch = xsimd::load_unaligned(&comatrix[i]);
        xsimd::batch_bool<float> condition = xsimd::fabs(comat_batch) > xsimd::batch<float>(eps);
        auto result_batch = xsimd::select(condition, xsimd::batch<float>(-1.0f) * comat_batch * xsimd::log2(comat_batch), xsimd::batch<float>(0.0f));
        result += xsimd::reduce_add(result_batch);
      }

      for (size_t i = algined_size; i < dim; i++)
      {
        if (std::fabs(comatrix[i]) > eps)
          result += -comatrix[i] * std::log2(comatrix[i]);
      }
    }
    break;

    default:
      break;
    }

    return result;
  }
}