#include "test_ncc.hpp"
#include "ncc.h"

#include <chrono>

TEST_F(TEST_NCC, CompareResults)
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
            auto ncc0 = ncc::ncc(ref0_frame.get(),
                                 test_case.ref0_stride,
                                 cur_frame.get(),
                                 test_case.cur_stride,
                                 test_case.width,
                                 test_case.height,
                                 32);

            auto ncc1 = ncc::ncc(ref1_frame.get(),
                                 test_case.ref1_stride,
                                 cur_frame.get(),
                                 test_case.cur_stride,
                                 test_case.width,
                                 test_case.height,
                                 32);
            org_time += Clock::now() - start;
            org_fps = (double)num_frame / org_time.count();

            start = Clock::now();
            auto ncc0_simd = ncc::ncc_simd(ref0_frame.get(),
                                           test_case.ref0_stride,
                                           cur_frame.get(),
                                           test_case.cur_stride,
                                           test_case.width,
                                           test_case.height,
                                           32);

            auto ncc1_simd = ncc::ncc_simd(ref1_frame.get(),
                                           test_case.ref1_stride,
                                           cur_frame.get(),
                                           test_case.cur_stride,
                                           test_case.width,
                                           test_case.height,
                                           32);
            simd_time += Clock::now() - start;
            simd_fps = (double)num_frame / simd_time.count();

            ASSERT_EQ(ncc0.size(), ncc0_simd.size());
            for (size_t i = 0; i < ncc0.size(); ++i)
            {
                EXPECT_NEAR(ncc0[i], ncc0_simd[i], 1e-3);
            }

            ASSERT_EQ(ncc1.size(), ncc1_simd.size());
            for (size_t i = 0; i < ncc1.size(); ++i)
            {
                EXPECT_NEAR(ncc1[i], ncc1_simd[i], 1e-3);
            }
        }
    }

    std::cout << "Original NCC Time: " << org_time.count() << " seconds, FPS: " << org_fps << " (" << num_frame << " frames)" << std::endl;
    std::cout << "SIMD     NCC Time: " << simd_time.count() << " seconds, FPS: " << simd_fps << " (" << num_frame << " frames)" << std::endl;
}