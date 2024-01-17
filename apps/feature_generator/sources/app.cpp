#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <queue>
#include <list>
#include <cassert>
#include <sstream>

#include "app.hpp"
#include "gop_dispenser.hpp"

#include "common_utils.h"
#include "thread_pool.hpp"
#include "program_options_lite.h"
#include "mathUtil.h"
#include "glcm.h"
#include "ncc.h"
#include "tc.h"
#include "yuv_reader.h"
#include "StopWatch.hpp"

#include "gtest/gtest.h"

int App::run(int argc, char **argv)
{
  if (!parse_config(argc, argv))
    return -1;

  keyint = std::max(int(keyframe_sec * input_fps + 0.5), 2 + bframes);
  int rem = (keyint - 2 - bframes) % (bframes + 1);
  if (rem != 0)
  {
    keyint += (bframes + 1 - rem);
  }

  derive_features();

  return 0;
}

void App::derive_features()
{
  yuv_reader yuv{};
  gop_dispenser disp{bframes, keyint};

  if (!yuv.open(input_yuv_path, input_width, input_height, 8))
  {
    exit(0);
  }

  picture_counts = 0;

  do
  {
    auto image = yuv.read_next();
    if (!image)
    {
      disp.terminate();
      break;
    }

    ++picture_counts;
    disp.put(image);
  } while (true);

  CThreadPool tp{1u};
  std::list<std::future<void>> lf;

  constexpr uint32_t num_glcm_features = glcm::NUM_PROPERTIES * glcm_angles * glcm_distances;
  constexpr double qpi = 3.14159265358979323846 / 4.0;

  glcm_feat.resize(picture_counts * num_glcm_features);
  ncc_feat.resize(picture_counts * 5);
  tc_feat.resize(picture_counts * 5);

  gop_output gout;

  while (disp.dequeue(gout))
  {
    for (int i = 0; i < gout.num_pictures; ++i)
    {
      int idx = gout.pictures[i].poc;
      assert(idx >= 0 && idx < (int)picture_counts);

      // GLCM
      for (int k = 0; k < glcm_angles * glcm_distances; ++k)
      {
        int dist = 1 + 2 * (k / glcm_angles);
        double angle = qpi * (k % glcm_angles);

        lf.emplace_back(tp.EnqueueJob(
            &App::derive_glcm, this,
            glcm_feat.data() + idx * num_glcm_features + k * glcm::NUM_PROPERTIES,
            gout.pictures[i].image,
            dist,
            angle));
      }

      // NCC
      lf.emplace_back(tp.EnqueueJob(
          &App::derive_ncc, this,
          ncc_feat.data() + idx * 5,
          gout.pictures[i].image,
          gout.pictures[i].ref_image0,
          gout.pictures[i].ref_image1));

      // TC
      lf.emplace_back(tp.EnqueueJob(
          &App::derive_tc, this,
          tc_feat.data() + idx * 5,
          gout.pictures[i].image,
          gout.pictures[i].ref_image0,
          gout.pictures[i].ref_image1));
    }
  }

  for (auto &f : lf)
  {
    f.wait();
  }
}

void App::derive_glcm(
    double *storage,
    std::shared_ptr<uint8_t> image,
    int glcm_distance,
    double glcm_angle)
{
  auto comatrix = glcm::graycomatrix(
      image.get(), input_width, 8, input_width, input_height, glcm_distance, glcm_angle, 8, true);

  for (int i = 0; i < glcm::NUM_PROPERTIES; ++i)
  {
    double value = glcm::graycoprops(comatrix, glcm::property(i));
    storage[i] = value;
  }
}

void App::derive_ncc(
    double *storage,
    std::shared_ptr<uint8_t> image,
    std::shared_ptr<uint8_t> ref_image0,
    std::shared_ptr<uint8_t> ref_image1)
{
  double a[5], b[5];

  if (ref_image0 != nullptr)
  {
    auto ncc0 = ncc::ncc(ref_image0.get(), input_width, image.get(), input_width, input_width, input_height, 32);
    a[0] = math_util::Mean(ncc0);
    a[1] = math_util::StandardDeviation(ncc0);
    a[2] = math_util::ShannonEntropy(ncc0);
    a[3] = math_util::Skewness(ncc0);
    a[4] = math_util::Kurtosis(ncc0);
  }

  if (ref_image1 != nullptr)
  {
    auto ncc1 = ncc::ncc(ref_image1.get(), input_width, image.get(), input_width, input_width, input_height, 32);
    b[0] = math_util::Mean(ncc1);
    b[1] = math_util::StandardDeviation(ncc1);
    b[2] = math_util::ShannonEntropy(ncc1);
    b[3] = math_util::Skewness(ncc1);
    b[4] = math_util::Kurtosis(ncc1);
  }

  if (ref_image0 == nullptr && ref_image1 == nullptr)
  {
    storage[0] = 1.0;
    storage[1] = storage[2] = storage[3] = storage[4] = 0.0;
  }
  else if (ref_image0 == nullptr)
  {
    std::memcpy(storage, b, sizeof(b));
  }
  else if (ref_image1 == nullptr)
  {
    std::memcpy(storage, a, sizeof(a));
  }
  else
  {
    double *p = a[0] > b[0] ? a : b;
    std::memcpy(storage, p, sizeof(a));
  }
}

void App::derive_tc(
    double *storage,
    std::shared_ptr<uint8_t> image,
    std::shared_ptr<uint8_t> ref_image0,
    std::shared_ptr<uint8_t> ref_image1)
{
  double a[5], b[5];

  if (ref_image0 != nullptr)
  {
    auto ncc0 = freq::dct8x8_tc_simd(ref_image0.get(), input_width, image.get(), input_width, input_width, input_height);
    a[0] = math_util::Mean(ncc0);
    a[1] = math_util::StandardDeviation(ncc0);
    a[2] = math_util::ShannonEntropy(ncc0);
    a[3] = math_util::Skewness(ncc0);
    a[4] = math_util::Kurtosis(ncc0);
  }

  if (ref_image1 != nullptr)
  {
    auto ncc1 = freq::dct8x8_tc_simd(ref_image1.get(), input_width, image.get(), input_width, input_width, input_height);
    b[0] = math_util::Mean(ncc1);
    b[1] = math_util::StandardDeviation(ncc1);
    b[2] = math_util::ShannonEntropy(ncc1);
    b[3] = math_util::Skewness(ncc1);
    b[4] = math_util::Kurtosis(ncc1);
  }

  if (ref_image0 == nullptr && ref_image1 == nullptr)
  {
    storage[0] = 1.0;
    storage[1] = storage[2] = storage[3] = storage[4] = 0.0;
  }
  else if (ref_image0 == nullptr)
  {
    std::memcpy(storage, b, sizeof(b));
  }
  else if (ref_image1 == nullptr)
  {
    std::memcpy(storage, a, sizeof(a));
  }
  else
  {
    double *p = a[0] > b[0] ? a : b;
    std::memcpy(storage, p, sizeof(a));
  }
}

bool App::parse_config(int argc, char **argv)
{
  namespace po = df::program_options_lite;

  bool do_help = false;

  po::Options opts;
  opts.addOptions()("help", do_help, false, "Print help text")("-i", input_yuv_path, std::string{}, "Input yuv file path")("-w", input_width, 0, "Input width")("-h", input_height, 0, "Input height")("-f", input_fps, 0.0, "Input fps")("-o", output_json_path, std::string{}, "Output json file path")("-tempdir", temp_dir_path, std::string{"./"}, "temp directory");

  po::setDefaults(opts);
  po::ErrorReporter err;

  const auto &argv_unhandled = po::scanArgv(opts, argc, (const char **)argv, err);

  for (auto a : argv_unhandled)
  {
    std::cerr << "Unhandled argument ignored: " << a << std::endl;
  }

  if (argc == 1 || do_help || input_yuv_path.empty() || output_json_path.empty())
  {
    po::doHelp(std::cout, opts);
    return false;
  }

  return true;
}