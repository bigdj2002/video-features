#pragma once

#include <memory>
#include <list>
#include <fstream>
#include <cstring>

#include "gtest/gtest.h"

class TEST_GLCM : public ::testing::Test
{
public:
  struct glcm_params
  {
    std::list<std::shared_ptr<uint8_t[]>> frames;
    int stride{0};
    int depth{0};
    int width{0};
    int height{0};
    int glcm_distance{0};
    double glcm_angle{0};
    int log2Levels{0};
    bool normalize{true};
  };

protected:
  std::vector<glcm_params> test_cases;
  std::ifstream file;

  static constexpr size_t frame_width{1920};
  static constexpr size_t frame_height{1080};

  void SetUp() override
  {
    // file.open("/mnt/sDisk2/jvet-test-sequences/ctc/sdr/BasketballDrive_1920x1080_50.yuv", std::ios::binary);
    file.open("/source/jvet-test-sequences/ctc/sdr/BasketballDrive_1920x1080_50.yuv", std::ios::binary);
    if (!file.is_open())
    {
      GTEST_SKIP() << "File open failed";
    }

    constexpr double qpi = 3.14159265358979323846 / 4.0;
    int frame_y_size = frame_width * frame_height;
    int frame_yuv_size = (frame_y_size * 3) >> 1; // Tested in only 420

    for (int k = 0; k < 12; ++k)
    {
      file.seekg(0, std::ios::beg);
      glcm_params params;

      while (true)
      {
        std::unique_ptr<uint8_t[]> frame(new uint8_t[frame_yuv_size]);
        file.read(reinterpret_cast<char *>(frame.get()), frame_yuv_size * sizeof(uint8_t));

        if ((std::size_t)file.gcount() < frame_yuv_size * sizeof(uint8_t))
        {
          break;
        }

        if (file.fail())
        {
          break;
        }

        std::unique_ptr<uint8_t[]> frame_y(new uint8_t[frame_y_size]);
        std::memcpy(frame_y.get(), frame.get(), frame_y_size * sizeof(uint8_t));
        params.frames.push_back(std::move(frame_y));
      }

      params.stride = frame_width;
      params.depth = 8;
      params.width = frame_width;
      params.height = frame_height;
      params.glcm_distance = 1 + 2 * (k / 4);
      params.glcm_angle = qpi * (k % 3);
      params.log2Levels = 8;
      params.normalize = true;
      test_cases.push_back(params);
    }
  }

  void TearDown() override
  {
    if (file.is_open())
    {
      file.close();
    }
  }
};