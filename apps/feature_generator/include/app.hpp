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

private:
  std::string input_yuv_path;
  int input_width;
  int input_height;
  double input_fps;
  std::string output_json_path;
  std::string temp_dir_path;

  static constexpr int bframes = 3;
  static constexpr double keyframe_sec = 1.0;
  int keyint;

  static constexpr int glcm_angles = 4;
  static constexpr int glcm_distances = 3;

  std::vector<double> glcm_feat, ncc_feat, tc_feat;
  uint32_t picture_counts = 0;

  static constexpr int QP_MIN = 22;
  static constexpr int QP_MAX = 51;
  static constexpr int NUM_QPS = QP_MAX - QP_MIN + 1;

  struct
  {
    std::vector<double> skip_ratio[NUM_QPS];
    std::vector<double> inter_ratio[NUM_QPS];
    std::vector<double> bpp[NUM_QPS];
    std::vector<double> avg_mvx[NUM_QPS];
    std::vector<double> avg_mvy[NUM_QPS];
    std::vector<double> satd[NUM_QPS];
    std::vector<double> mse[NUM_QPS];
    std::vector<double> vmaf[NUM_QPS];
    std::vector<char> pic_type[NUM_QPS];
  } stat;

  std::vector<int> dec_to_disp;

};