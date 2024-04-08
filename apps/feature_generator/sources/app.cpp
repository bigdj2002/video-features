#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <queue>
#include <list>
#include <cassert>
#include <sstream>

#include "app.hpp"
#include "gop_distributor.hpp"

#include "common_utils.h"
#include "thread_pool.hpp"
#include "program_options_lite.h"
#include "mathUtil.h"
#include "glcm.h"
#include "ncc.h"
#include "tc.h"
#include "nlp.h"
#include "yuv_reader.h"
#include "StopWatch.hpp"

#include "gtest/gtest.h"
#include "jsoncpp/json/json.h"

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

  save_as_json();

  return 0;
}

void App::derive_features()
{
  yuv_reader yuv{};
  gop_distributor disp{bframes, keyint};

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

  CThreadPool tp{(unsigned)num_threads};
  std::list<std::future<void>> lf;

  constexpr uint32_t num_glcm_features = glcm::NUM_PROPERTIES * glcm_angles * glcm_distances;
  constexpr double qpi = 3.14159265358979323846 / 4.0;

  glcm_feat.resize(picture_counts * num_glcm_features);
  ncc_feat.resize(picture_counts * 5);
  tc_feat.resize(picture_counts * 5);
  nlp_feat.resize(picture_counts * 5);

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

      // NLP
      lf.emplace_back(tp.EnqueueJob(
          &App::derive_nlp, this,
          nlp_feat.data() + idx * 5,
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
  if (enable_simd)
  {
    auto comatrix = glcm::graycomatrix_simd(image.get(), input_width, 8, input_width, input_height, glcm_distance, glcm_angle, 8, true);
    for (int i = 0; i < glcm::NUM_PROPERTIES; ++i)
    {
      double value = glcm::graycoprops_simd(comatrix, glcm::property(i));
      storage[i] = value;
    }
  }
  else
  {
    auto comatrix = glcm::graycomatrix(image.get(), input_width, 8, input_width, input_height, glcm_distance, glcm_angle, 8, true);
    for (int i = 0; i < glcm::NUM_PROPERTIES; ++i)
    {
      double value = glcm::graycoprops(comatrix, glcm::property(i));
      storage[i] = value;
    }
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
    if (enable_simd)
    {
      auto ncc0 = ncc::ncc_simd(ref_image0.get(), input_width, image.get(), input_width, input_width, input_height, processed_blk_size);
      a[0] = math_util::Mean_simd(ncc0);
      a[1] = math_util::StandardDeviation_simd(ncc0);
      a[2] = math_util::ShannonEntropy_simd(ncc0);
      a[3] = math_util::Skewness_simd(ncc0);
      a[4] = math_util::Kurtosis_simd(ncc0);
    }
    else
    {
      auto ncc0 = ncc::ncc(ref_image0.get(), input_width, image.get(), input_width, input_width, input_height, processed_blk_size);
      a[0] = math_util::Mean(ncc0);
      a[1] = math_util::StandardDeviation(ncc0);
      a[2] = math_util::ShannonEntropy(ncc0);
      a[3] = math_util::Skewness(ncc0);
      a[4] = math_util::Kurtosis(ncc0);
    }
  }

  if (ref_image1 != nullptr)
  {
    if (enable_simd)
    {
      auto ncc1 = ncc::ncc_simd(ref_image1.get(), input_width, image.get(), input_width, input_width, input_height, processed_blk_size);
      b[0] = math_util::Mean_simd(ncc1);
      b[1] = math_util::StandardDeviation_simd(ncc1);
      b[2] = math_util::ShannonEntropy_simd(ncc1);
      b[3] = math_util::Skewness_simd(ncc1);
      b[4] = math_util::Kurtosis_simd(ncc1);
    }
    else
    {
      auto ncc1 = ncc::ncc(ref_image1.get(), input_width, image.get(), input_width, input_width, input_height, processed_blk_size);
      b[0] = math_util::Mean(ncc1);
      b[1] = math_util::StandardDeviation(ncc1);
      b[2] = math_util::ShannonEntropy(ncc1);
      b[3] = math_util::Skewness(ncc1);
      b[4] = math_util::Kurtosis(ncc1);
    }
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
    if (enable_simd)
    {
      auto ncc0 = freq::dct8x8_tc_simd(ref_image0.get(), input_width, image.get(), input_width, input_width, input_height);
      a[0] = math_util::Mean_simd(ncc0);
      a[1] = math_util::StandardDeviation_simd(ncc0);
      a[2] = math_util::ShannonEntropy_simd(ncc0);
      a[3] = math_util::Skewness_simd(ncc0);
      a[4] = math_util::Kurtosis_simd(ncc0);
    }
    else
    {
      auto ncc0 = freq::dct8x8_tc(ref_image0.get(), input_width, image.get(), input_width, input_width, input_height);
      a[0] = math_util::Mean(ncc0);
      a[1] = math_util::StandardDeviation(ncc0);
      a[2] = math_util::ShannonEntropy(ncc0);
      a[3] = math_util::Skewness(ncc0);
      a[4] = math_util::Kurtosis(ncc0);
    }
  }

  if (ref_image1 != nullptr)
  {
    if (enable_simd)
    {
      auto ncc1 = freq::dct8x8_tc_simd(ref_image1.get(), input_width, image.get(), input_width, input_width, input_height);
      b[0] = math_util::Mean_simd(ncc1);
      b[1] = math_util::StandardDeviation_simd(ncc1);
      b[2] = math_util::ShannonEntropy_simd(ncc1);
      b[3] = math_util::Skewness_simd(ncc1);
      b[4] = math_util::Kurtosis_simd(ncc1);
    }
    else
    {
      auto ncc1 = freq::dct8x8_tc(ref_image1.get(), input_width, image.get(), input_width, input_width, input_height);
      b[0] = math_util::Mean(ncc1);
      b[1] = math_util::StandardDeviation(ncc1);
      b[2] = math_util::ShannonEntropy(ncc1);
      b[3] = math_util::Skewness(ncc1);
      b[4] = math_util::Kurtosis(ncc1);
    }
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

void App::derive_nlp(
    double *storage,
    std::shared_ptr<uint8_t> image,
    std::shared_ptr<uint8_t> ref_image0,
    std::shared_ptr<uint8_t> ref_image1)
{
  double a[5], b[5];

  if (ref_image0 != nullptr)
  {
    // if (enable_simd) // TODO: Not implemented yet
    if (0)
    {
      auto nlp0 = nlp::nlp_simd(ref_image0.get(), input_width, image.get(), input_width, input_width, input_height, processed_blk_size);
      a[0] = math_util::Mean_simd(nlp0);
      a[1] = math_util::StandardDeviation_simd(nlp0);
      a[2] = math_util::ShannonEntropy_simd(nlp0);
      a[3] = math_util::Skewness_simd(nlp0);
      a[4] = math_util::Kurtosis_simd(nlp0);
    }
    else
    {
      auto nlp0 = nlp::nlp(ref_image0.get(), input_width, image.get(), input_width, input_width, input_height, processed_blk_size);
      a[0] = math_util::Mean(nlp0);
      a[1] = math_util::StandardDeviation(nlp0);
      a[2] = math_util::ShannonEntropy(nlp0);
      a[3] = math_util::Skewness(nlp0);
      a[4] = math_util::Kurtosis(nlp0);
    }
  }

  if (ref_image1 != nullptr)
  {
    // if (enable_simd) // TODO: Not implemented yet
    if (0)
    {
      auto nlp1 = nlp::nlp_simd(ref_image1.get(), input_width, image.get(), input_width, input_width, input_height, processed_blk_size);
      b[0] = math_util::Mean_simd(nlp1);
      b[1] = math_util::StandardDeviation_simd(nlp1);
      b[2] = math_util::ShannonEntropy_simd(nlp1);
      b[3] = math_util::Skewness_simd(nlp1);
      b[4] = math_util::Kurtosis_simd(nlp1);
    }
    else
    {
      auto nlp1 = nlp::nlp(ref_image1.get(), input_width, image.get(), input_width, input_width, input_height, processed_blk_size);
      b[0] = math_util::Mean(nlp1);
      b[1] = math_util::StandardDeviation(nlp1);
      b[2] = math_util::ShannonEntropy(nlp1);
      b[3] = math_util::Skewness(nlp1);
      b[4] = math_util::Kurtosis(nlp1);
    }
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
  opts.addOptions()
  ("help", do_help, false, "Print help text")
  ("-i", input_yuv_path, std::string{}, "Input yuv file path")
  ("-w", input_width, 1920, "Input width (Default: 1920)")
  ("-h", input_height, 1080, "Input height (Default: 1080)")
  ("-f", input_fps, 30.0, "Input fps (Default: 30)")
  ("-b", processed_blk_size, 32, "Processed block size (Default: 32)")
  ("-t", num_threads, 40, "Number of thread used (Default: 40)")
  ("-s", enable_simd, 0, "Enabling SIMD (Default: 0)")
  ("-o", output_json_path, std::string{}, "Output json file path");

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

void App::save_as_json()
{
  constexpr int num_glcm_features = glcm::NUM_PROPERTIES * glcm_angles * glcm_distances;

  Json::Value root;
  Json::Value frames = Json::Value{Json::arrayValue};

  for (uint32_t i = 0; i < picture_counts; ++i)
  {
    Json::Value frame;

    Json::Value glcm;
    for (int k = 0; k < num_glcm_features; k++)
    {
      glcm[k] = glcm_feat[i * num_glcm_features + k];
    }
    frame["glcm"] = glcm;

    Json::Value ncc;
    for (int k = 0; k < 5; k++)
    {
      ncc[k] = ncc_feat[i * 5 + k];
    }
    frame["ncc"] = ncc;

    Json::Value tc;
    for (int k = 0; k < 5; k++)
    {
      tc[k] = tc_feat[i * 5 + k];
    }
    frame["tc"] = tc;

    Json::Value nlp;
    for (int k = 0; k < 5; k++)
    {
      nlp[k] = nlp_feat[i * 5 + k];
    }
    frame["nlp"] = nlp;

    frames.append(frame);
  }
  root["frames"] = frames;

  std::ofstream f{output_json_path};
  f << root;
  f.close();
}
