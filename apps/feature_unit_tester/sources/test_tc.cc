#include "test_tc.hpp"
#include "tc.h"

#include <chrono>

TEST_F(TEST_TC, CompareResults)
{
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::duration<double> org_time(0), simd_time(0);
    double org_fps{0}, simd_fps{0};
    int num_frame = test_cases[0].cur_frames.size();

    for (auto &test_case : test_cases)
    {
        while (!test_case.cur_frames.empty())
        {
            std::shared_ptr<uint8_t[]> cur_frame = test_case.cur_frames.front();
            test_case.cur_frames.pop_front();
            std::shared_ptr<uint8_t[]> ref0_frame = test_case.ref0_frames.front();
            test_case.ref0_frames.pop_front();
            std::shared_ptr<uint8_t[]> ref1_frame = test_case.ref1_frames.front();
            test_case.ref1_frames.pop_front();

            auto start = Clock::now();
            auto tc0 = freq::dct8x8_tc(ref0_frame.get(),
                                       test_case.ref0_stride,
                                       cur_frame.get(),
                                       test_case.cur_stride,
                                       test_case.width,
                                       test_case.height);
            auto tc1 = freq::dct8x8_tc(ref1_frame.get(),
                                       test_case.ref1_stride,
                                       cur_frame.get(),
                                       test_case.cur_stride,
                                       test_case.width,
                                       test_case.height);
            org_time += Clock::now() - start;
            org_fps = (double)num_frame / org_time.count();

            start = Clock::now();
            auto tc0_simd = freq::dct8x8_tc_simd(ref0_frame.get(),
                                                 test_case.ref0_stride,
                                                 cur_frame.get(),
                                                 test_case.cur_stride,
                                                 test_case.width,
                                                 test_case.height);
            auto tc1_simd = freq::dct8x8_tc_simd(ref1_frame.get(),
                                                 test_case.ref1_stride,
                                                 cur_frame.get(),
                                                 test_case.cur_stride,
                                                 test_case.width,
                                                 test_case.height);
            simd_time += Clock::now() - start;
            simd_fps = (double)num_frame / simd_time.count();

            ASSERT_EQ(tc0.size(), tc0_simd.size());
            for (size_t i = 0; i < tc0.size(); ++i)
            {
                EXPECT_NEAR(tc0[i], tc0_simd[i], 1e-3);
            }

            ASSERT_EQ(tc1.size(), tc1_simd.size());
            for (size_t i = 0; i < tc1.size(); ++i)
            {
                EXPECT_NEAR(tc1[i], tc1_simd[i], 1e-3);
            }
        }
    }

    std::cout << "Original TC Time: " << org_time.count() << " seconds, FPS: " << org_fps << " (" << num_frame << " frames)" << std::endl;
    std::cout << "SIMD     TC Time: " << simd_time.count() << " seconds, FPS: " << simd_fps << " (" << num_frame << " frames)" << std::endl;
}