#include "test_glcm.hpp"
#include "glcm.h"

#include <chrono>

TEST_F(TEST_GLCM, CompareResults)
{
  using Clock = std::chrono::high_resolution_clock;
  std::chrono::duration<double> org_time(0), simd_time(0);
  double org_fps{0}, simd_fps{0};
  int num_frame = test_cases[0].frames.size();

  for (auto &test_case : test_cases)
  {
    while (!test_case.frames.empty())
    {
      std::shared_ptr<uint8_t[]> frame = test_case.frames.front();
      test_case.frames.pop_front();

      auto start = Clock::now();
      auto comatrix = glcm::graycomatrix(frame.get(),
                                         test_case.stride,
                                         test_case.depth,
                                         test_case.width,
                                         test_case.height,
                                         test_case.glcm_distance,
                                         test_case.glcm_angle,
                                         test_case.log2Levels,
                                         test_case.normalize);
      double org_value[glcm::NUM_PROPERTIES];
      for (int i = 0; i < glcm::NUM_PROPERTIES; ++i)
        org_value[i] = glcm::graycoprops(comatrix, glcm::property(i));
      org_time += Clock::now() - start;
      org_fps = (double)num_frame / org_time.count();

      start = Clock::now();
      auto comatrix_simd = glcm::graycomatrix_simd(frame.get(),
                                                   test_case.stride,
                                                   test_case.depth,
                                                   test_case.width,
                                                   test_case.height,
                                                   test_case.glcm_distance,
                                                   test_case.glcm_angle,
                                                   test_case.log2Levels,
                                                   test_case.normalize);
      double simd_value[glcm::NUM_PROPERTIES];
      for (int i = 0; i < glcm::NUM_PROPERTIES; ++i)
        simd_value[i] = glcm::graycoprops(comatrix, glcm::property(i));
      simd_time += Clock::now() - start;
      simd_fps = (double)num_frame / simd_time.count();

      ASSERT_EQ(comatrix.size(), comatrix_simd.size());
      for (size_t i = 0; i < comatrix.size(); ++i)
      {
        EXPECT_NEAR(comatrix[i], comatrix_simd[i], 1e-6);
      }

      for (size_t i = 0; i < glcm::NUM_PROPERTIES; ++i)
      {
        EXPECT_NEAR(org_value[i], simd_value[i], 1e-2);
      }
    }
  }

  std::cout << "Original GLCM Time: " << org_time.count() << " seconds, FPS: " << org_fps << " (" << num_frame << " frames)" << std::endl;
  std::cout << "SIMD     GLCM Time: " << simd_time.count() << " seconds, FPS: " << simd_fps << " (" << num_frame << " frames)" << std::endl;
}