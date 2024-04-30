#pragma once

#include <string>
#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>

class App
{
public:
  int run(int argc, char **argv);

private:
  bool parse_config(int argc, char** argv);
  void derive_features();

  void derive_glcm(
      double *storage,
      std::shared_ptr<uint8_t> image,
      int glcm_distance,
      double glcm_angle);

  void derive_ncc(
      double *storage,
      std::shared_ptr<uint8_t> image,
      std::shared_ptr<uint8_t> ref_image0,
      std::shared_ptr<uint8_t> ref_image1);

  void derive_tc(
      double *storage,
      std::shared_ptr<uint8_t> image,
      std::shared_ptr<uint8_t> ref_image0,
      std::shared_ptr<uint8_t> ref_image1);

  void derive_nlp(
      double *storage,
      std::shared_ptr<uint8_t> image,
      std::shared_ptr<uint8_t> ref_image0,
      std::shared_ptr<uint8_t> ref_image1);

  void derive_pca();
  void save_as_json();

private:
  std::string input_yuv_path;
  int input_width;
  int input_height;
  double input_fps;
  int processed_blk_size;
  int pca_output_num;
  int num_threads;
  int enable_simd;
  std::string output_json_path;

  double keyframe_sec;
  int keyint;

  static constexpr int glcm_angles = 4;
  static constexpr int glcm_distances = 5;

  std::vector<double> glcm_feat, ncc_feat, tc_feat, nlp_feat, pca_feat;
  uint32_t picture_counts = 0;
};