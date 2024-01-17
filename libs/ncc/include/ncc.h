#pragma once

#include <vector>
#include <cstdint>

namespace ncc
{
  extern std::vector<double> ncc(
      const uint8_t *ref_image,
      uint32_t ref_stride,
      const uint8_t *tar_image,
      uint32_t tar_stride,
      uint32_t image_width,
      uint32_t image_height,
      uint32_t block_size,
      int ddof = 1);

  extern std::vector<float> ncc_simd(
      const uint8_t *ref_image,
      uint32_t ref_stride,
      const uint8_t *tar_image,
      uint32_t tar_stride,
      uint32_t image_width,
      uint32_t image_height,
      uint32_t block_size,
      int ddof = 1);
}