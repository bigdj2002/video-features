#pragma once

#include <memory>
#include <list>
#include <fstream>
#include <cstring>

#include "gtest/gtest.h"

class TEST_NCC : public ::testing::Test
{
public:
  struct ncc_params
  {
    std::list<std::shared_ptr<uint8_t[]>> ref0_frames;
    int ref0_stride{0};
    std::list<std::shared_ptr<uint8_t[]>> ref1_frames;
    int ref1_stride{0};
    std::list<std::shared_ptr<uint8_t[]>> cur_frames;
    int cur_stride{0};
    int width{0};
    int height{0};
  };

protected:
  std::vector<ncc_params> test_cases;
  std::ifstream file;

  static constexpr size_t frame_width{1920};
  static constexpr size_t frame_height{1080};

  void SetUp() override
  {
    // file.open("/mnt/sDisk2/jvet-test-sequences/ctc/sdr/BasketballDrive_1920x1080_50.yuv", std::ios::binary);
    file.open("/mnt/sDisk2/jvet-test-sequences/ctc/sdr/BasketballDrive_1920x1080_50.yuv", std::ios::binary);
    if (!file.is_open())
    {
      GTEST_SKIP() << "File open failed";
    }

    ncc_params params;
    int frame_y_size = frame_width * frame_height;
    int frame_yuv_size = (frame_y_size * 3) >> 1; // Tested in only 420
    int gop_size = frame_yuv_size * 3;

    while (true)
    {
      std::unique_ptr<uint8_t[]> frames(new uint8_t[gop_size]);
      file.read(reinterpret_cast<char *>(frames.get()), gop_size * sizeof(uint8_t));

      if ((std::size_t)file.gcount() < gop_size * sizeof(uint8_t))
      {
        break;
      }

      if (file.fail())
      {
        break;
      }

      std::unique_ptr<uint8_t[]> ref0_frame(new uint8_t[frame_y_size]);
      std::memcpy(ref0_frame.get(), frames.get(), frame_y_size);
      params.ref0_frames.push_back(std::move(ref0_frame));

      std::unique_ptr<uint8_t[]> cur_frame(new uint8_t[frame_y_size]);
      std::memcpy(cur_frame.get(), frames.get() + frame_yuv_size, frame_y_size);
      params.cur_frames.push_back(std::move(cur_frame));

      std::unique_ptr<uint8_t[]> ref1_frame(new uint8_t[frame_y_size]);
      std::memcpy(ref1_frame.get(), frames.get() + (frame_yuv_size * 2), frame_y_size);
      params.ref1_frames.push_back(std::move(ref1_frame));
    }

    params.ref0_stride = frame_width;
    params.ref1_stride = frame_width;
    params.cur_stride = frame_width;
    params.width = frame_width;
    params.height = frame_height;
    test_cases.push_back(params);
  }

  void TearDown() override
  {
    if (file.is_open())
    {
      file.close();
    }
  }
};