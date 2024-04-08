#pragma once

#include <vector>
#include <cstdint>

namespace glcm
{
  std::vector<double> graycomatrix(
      const uint8_t *src_luma,
      int src_stride,
      int src_depth,
      int src_width,
      int src_height,
      int distance,
      double angle,
      int log2Levels,
      bool normalize);

  std::vector<float> graycomatrix_simd(
      const uint8_t *src_luma,
      int src_stride,
      int src_depth,
      int src_width,
      int src_height,
      int distance,
      double angle,
      int log2Levels,
      bool normalize);

  enum property
  {
    dissimilarity = 0,
    contrast,
    homogeneity,
    energy,
    correlation,
    entropy,
    NUM_PROPERTIES
  };

  double graycoprops(
      const std::vector<double> &comatrix,
      property prop);

  double graycoprops_simd(
      const std::vector<float> &comatrix,
      property prop);
}
