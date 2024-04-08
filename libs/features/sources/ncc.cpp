#include <cstring>
#include <stdexcept>
#include <limits>

#include "ncc.h"
#include "mathUtil.h"

#include "xsimd.hpp"

static void as_contiguous_memory(
    uint8_t *dst,
    const uint8_t *src,
    size_t src_stride,
    size_t width,
    size_t height)
{
  for (size_t i = 0; i < height; i++)
  {
    std::memcpy(&dst[width * i], &src[src_stride * i], width);
  }
}

static void as_contiguous_memory_simd(
    uint8_t *dst,
    const uint8_t *src,
    size_t src_stride,
    size_t width,
    size_t height)
{
  for (size_t i = 0; i < height; i++)
  {
    math_util::xsimd_memcpy<uint8_t>(&dst[width * i], &src[src_stride * i], width);
  }
}

static std::vector<double> normalize(
    const std::vector<uint8_t> &v,
    int ddof)
{
  if (v.size() - ddof <= 0)
    std::invalid_argument("Shoud be satisfied \"ref.size() - ddof > 0\"");

  double stdvar = math_util::StandardDeviation(v, ddof);
  double mean = double(math_util::Accumulation(v)) / v.size();

  const double eps = std::numeric_limits<double>::epsilon();
  uint32_t n = v.size();

  if (std::abs(stdvar) < eps)
    stdvar = 1e-6;

  std::vector<double> result(v.size());
  for (uint32_t i = 0; i < n; i++)
  {
    result[i] = (v[i] - mean) / stdvar;
  }

  return result;
}

static std::vector<float> normalize_simd(
    const std::vector<uint8_t> &v,
    int ddof)
{
  std::vector<float> v_temp(v.size());
  std::transform(v.begin(), v.end(), v_temp.begin(), [](uint8_t val)
                 { return static_cast<float>(val); });

  if (v_temp.size() - ddof <= 0)
    std::invalid_argument("Shoud be satisfied \"ref.size() - ddof > 0\"");

  float stdvar = math_util::StandardDeviation_simd(v_temp, ddof);
  float mean = float(math_util::Accumulation_simd(v_temp)) / v_temp.size();

  const float eps = std::numeric_limits<float>::epsilon();
  uint32_t n = v_temp.size();

  if (std::abs(stdvar) < eps)
    stdvar = 1e-6;

  std::vector<float> result(v_temp.size());

  using batch_type = xsimd::batch<float>;
  const size_t batch_size = batch_type::size;
  const size_t algined_size = n - n % batch_size;

  for (uint32_t i = 0; i < algined_size; i += batch_size)
  {
    auto v_batch = xsimd::batch_cast<float>(xsimd::load_unaligned(&v_temp[i]));
    auto result_batch = xsimd::div(xsimd::sub(v_batch, batch_type(mean)), batch_type(stdvar));
    xsimd::store_unaligned(&result[i], result_batch);
  }

  for (uint32_t i = algined_size; i < n; i++)
  {
    result[i] = (v_temp[i] - mean) / stdvar;
  }

  return result;
}

static double compute_ncc(
    const std::vector<uint8_t> &ref,
    const std::vector<uint8_t> &tar,
    int ddof)
{
  auto nref = normalize(ref, ddof);
  auto ntar = normalize(tar, ddof);

  return (1.0 / (ref.size() - ddof)) * math_util::InnerProduct(nref, ntar);
}

static float compute_ncc_simd(
    const std::vector<uint8_t> &ref,
    const std::vector<uint8_t> &tar,
    int ddof)
{
  auto nref = normalize_simd(ref, ddof);
  auto ntar = normalize_simd(tar, ddof);

  return (1.0 / (ref.size() - ddof)) * math_util::InnerProduct_simd(nref, ntar);
}


namespace ncc
{
  std::vector<double> ncc(
      const uint8_t *ref_image,
      uint32_t ref_stride,
      const uint8_t *tar_image,
      uint32_t tar_stride,
      uint32_t image_width,
      uint32_t image_height,
      uint32_t block_size,
      int ddof)
  {
    const uint32_t nblkx = image_width / block_size;
    const uint32_t nblky = image_height / block_size;

    std::vector<double> result(nblkx * nblky);
    std::vector<uint8_t> ref_cache(block_size * block_size);
    std::vector<uint8_t> tar_cache(block_size * block_size);
    size_t idx = 0;

    for (unsigned by = 0; by < nblky; by++)
    {
      for (unsigned bx = 0; bx < nblkx; bx++, idx++)
      {
        int x = bx * block_size;
        int y = by * block_size;

        as_contiguous_memory(ref_cache.data(), &ref_image[y * ref_stride + x], ref_stride, block_size, block_size);
        as_contiguous_memory(tar_cache.data(), &tar_image[y * tar_stride + x], tar_stride, block_size, block_size);

        double value = compute_ncc(ref_cache, tar_cache, ddof);
        if (value > 1.0 || value < -1.0)
        {
          __attribute__((unused)) bool stop = true;
          __attribute__((unused)) double value = compute_ncc(ref_cache, tar_cache, ddof);
        }
        result[idx] = value;
      }
    }

    return result;
  }

  std::vector<float> ncc_simd(
      const uint8_t *ref_image,
      uint32_t ref_stride,
      const uint8_t *tar_image,
      uint32_t tar_stride,
      uint32_t image_width,
      uint32_t image_height,
      uint32_t block_size,
      int ddof)
  {
    const uint32_t nblkx = image_width / block_size;
    const uint32_t nblky = image_height / block_size;

    std::vector<float> result(nblkx * nblky);
    std::vector<uint8_t> ref_cache(block_size * block_size);
    std::vector<uint8_t> tar_cache(block_size * block_size);
    size_t idx = 0;

    for (unsigned by = 0; by < nblky; by++)
    {
      for (unsigned bx = 0; bx < nblkx; bx++, idx++)
      {
        int x = bx * block_size;
        int y = by * block_size;

        as_contiguous_memory_simd(ref_cache.data(), &ref_image[y * ref_stride + x], ref_stride, block_size, block_size);
        as_contiguous_memory_simd(tar_cache.data(), &tar_image[y * tar_stride + x], tar_stride, block_size, block_size);

        float value = compute_ncc_simd(ref_cache, tar_cache, ddof);
        if (value > 1.0 || value < -1.0)
        {
          __attribute__((unused)) bool stop = true;
          __attribute__((unused)) float value = compute_ncc_simd(ref_cache, tar_cache, ddof);
        }
        result[idx] = value;
      }
    }

    return result;
  }
}