#pragma once

#include <fftw3.h>
#include <complex>
#include <vector>
#include <cstdint>
#include <string>

#include <immintrin.h>
#include "xsimd.hpp"

#ifdef __AVX2__
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

#define DEFINE_DCT2_P8_MATRIX(a, b, c, d, e, f, g) \
  {                                                \
    {a, a, a, a, a, a, a, a},                      \
        {d, e, f, g, -g, -f, -e, -d},              \
        {b, c, -c, -b, -b, -c, c, b},              \
        {e, -g, -d, -f, f, d, g, -e},              \
        {a, -a, -a, a, a, -a, -a, a},              \
        {f, -d, g, e, -e, -g, d, -f},              \
        {c, -b, b, -c, -c, b, -b, c},              \
    {                                              \
      g, -f, e, -d, d, -e, f, -g                   \
    }                                              \
  }

#define DEFINE_DCT2_P16_MATRIX(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) \
  {                                                                         \
    {a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a},                       \
        {h, i, j, k, l, m, n, o, -o, -n, -m, -l, -k, -j, -i, -h},           \
        {d, e, f, g, -g, -f, -e, -d, -d, -e, -f, -g, g, f, e, d},           \
        {i, l, o, -m, -j, -h, -k, -n, n, k, h, j, m, -o, -l, -i},           \
        {b, c, -c, -b, -b, -c, c, b, b, c, -c, -b, -b, -c, c, b},           \
        {j, o, -k, -i, -n, l, h, m, -m, -h, -l, n, i, k, -o, -j},           \
        {e, -g, -d, -f, f, d, g, -e, -e, g, d, f, -f, -d, -g, e},           \
        {k, -m, -i, o, h, n, -j, -l, l, j, -n, -h, -o, i, m, -k},           \
        {a, -a, -a, a, a, -a, -a, a, a, -a, -a, a, a, -a, -a, a},           \
        {l, -j, -n, h, -o, -i, m, k, -k, -m, i, o, -h, n, j, -l},           \
        {f, -d, g, e, -e, -g, d, -f, -f, d, -g, -e, e, g, -d, f},           \
        {m, -h, l, n, -i, k, o, -j, j, -o, -k, i, -n, -l, h, -m},           \
        {c, -b, b, -c, -c, b, -b, c, c, -b, b, -c, -c, b, -b, c},           \
        {n, -k, h, -j, m, o, -l, i, -i, l, -o, -m, j, -h, k, -n},           \
        {g, -f, e, -d, d, -e, f, -g, -g, f, -e, d, -d, e, -f, g},           \
    {                                                                       \
      o, -n, m, -l, k, -j, i, -h, h, -i, j, -k, l, -m, n, -o                \
    }                                                                       \
  }

#define DEFINE_DCT2_P32_MATRIX(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E) \
  {                                                                                                                         \
    {a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a},                       \
        {p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, -E, -D, -C, -B, -A, -z, -y, -x, -w, -v, -u, -t, -s, -r, -q, -p},   \
        {h, i, j, k, l, m, n, o, -o, -n, -m, -l, -k, -j, -i, -h, -h, -i, -j, -k, -l, -m, -n, -o, o, n, m, l, k, j, i, h},   \
        {q, t, w, z, C, -E, -B, -y, -v, -s, -p, -r, -u, -x, -A, -D, D, A, x, u, r, p, s, v, y, B, E, -C, -z, -w, -t, -q},   \
        {d, e, f, g, -g, -f, -e, -d, -d, -e, -f, -g, g, f, e, d, d, e, f, g, -g, -f, -e, -d, -d, -e, -f, -g, g, f, e, d},   \
        {r, w, B, -D, -y, -t, -p, -u, -z, -E, A, v, q, s, x, C, -C, -x, -s, -q, -v, -A, E, z, u, p, t, y, D, -B, -w, -r},   \
        {i, l, o, -m, -j, -h, -k, -n, n, k, h, j, m, -o, -l, -i, -i, -l, -o, m, j, h, k, n, -n, -k, -h, -j, -m, o, l, i},   \
        {s, z, -D, -w, -p, -v, -C, A, t, r, y, -E, -x, -q, -u, -B, B, u, q, x, E, -y, -r, -t, -A, C, v, p, w, D, -z, -s},   \
        {b, c, -c, -b, -b, -c, c, b, b, c, -c, -b, -b, -c, c, b, b, c, -c, -b, -b, -c, c, b, b, c, -c, -b, -b, -c, c, b},   \
        {t, C, -y, -p, -x, D, u, s, B, -z, -q, -w, E, v, r, A, -A, -r, -v, -E, w, q, z, -B, -s, -u, -D, x, p, y, -C, -t},   \
        {j, o, -k, -i, -n, l, h, m, -m, -h, -l, n, i, k, -o, -j, -j, -o, k, i, n, -l, -h, -m, m, h, l, -n, -i, -k, o, j},   \
        {u, -E, -t, -v, D, s, w, -C, -r, -x, B, q, y, -A, -p, -z, z, p, A, -y, -q, -B, x, r, C, -w, -s, -D, v, t, E, -u},   \
        {e, -g, -d, -f, f, d, g, -e, -e, g, d, f, -f, -d, -g, e, e, -g, -d, -f, f, d, g, -e, -e, g, d, f, -f, -d, -g, e},   \
        {v, -B, -p, -C, u, w, -A, -q, -D, t, x, -z, -r, -E, s, y, -y, -s, E, r, z, -x, -t, D, q, A, -w, -u, C, p, B, -v},   \
        {k, -m, -i, o, h, n, -j, -l, l, j, -n, -h, -o, i, m, -k, -k, m, i, -o, -h, -n, j, l, -l, -j, n, h, o, -i, -m, k},   \
        {w, -y, -u, A, s, -C, -q, E, p, D, -r, -B, t, z, -v, -x, x, v, -z, -t, B, r, -D, -p, -E, q, C, -s, -A, u, y, -w},   \
        {a, -a, -a, a, a, -a, -a, a, a, -a, -a, a, a, -a, -a, a, a, -a, -a, a, a, -a, -a, a, a, -a, -a, a, a, -a, -a, a},   \
        {x, -v, -z, t, B, -r, -D, p, -E, -q, C, s, -A, -u, y, w, -w, -y, u, A, -s, -C, q, E, -p, D, r, -B, -t, z, v, -x},   \
        {l, -j, -n, h, -o, -i, m, k, -k, -m, i, o, -h, n, j, -l, -l, j, n, -h, o, i, -m, -k, k, m, -i, -o, h, -n, -j, l},   \
        {y, -s, -E, r, -z, -x, t, D, -q, A, w, -u, -C, p, -B, -v, v, B, -p, C, u, -w, -A, q, -D, -t, x, z, -r, E, s, -y},   \
        {f, -d, g, e, -e, -g, d, -f, -f, d, -g, -e, e, g, -d, f, f, -d, g, e, -e, -g, d, -f, -f, d, -g, -e, e, g, -d, f},   \
        {z, -p, A, y, -q, B, x, -r, C, w, -s, D, v, -t, E, u, -u, -E, t, -v, -D, s, -w, -C, r, -x, -B, q, -y, -A, p, -z},   \
        {m, -h, l, n, -i, k, o, -j, j, -o, -k, i, -n, -l, h, -m, -m, h, -l, -n, i, -k, -o, j, -j, o, k, -i, n, l, -h, m},   \
        {A, -r, v, -E, -w, q, -z, -B, s, -u, D, x, -p, y, C, -t, t, -C, -y, p, -x, -D, u, -s, B, z, -q, w, E, -v, r, -A},   \
        {c, -b, b, -c, -c, b, -b, c, c, -b, b, -c, -c, b, -b, c, c, -b, b, -c, -c, b, -b, c, c, -b, b, -c, -c, b, -b, c},   \
        {B, -u, q, -x, E, y, -r, t, -A, -C, v, -p, w, -D, -z, s, -s, z, D, -w, p, -v, C, A, -t, r, -y, -E, x, -q, u, -B},   \
        {n, -k, h, -j, m, o, -l, i, -i, l, -o, -m, j, -h, k, -n, -n, k, -h, j, -m, -o, l, -i, i, -l, o, m, -j, h, -k, n},   \
        {C, -x, s, -q, v, -A, -E, z, -u, p, -t, y, -D, -B, w, -r, r, -w, B, D, -y, t, -p, u, -z, E, A, -v, q, -s, x, -C},   \
        {g, -f, e, -d, d, -e, f, -g, -g, f, -e, d, -d, e, -f, g, g, -f, e, -d, d, -e, f, -g, -g, f, -e, d, -d, e, -f, g},   \
        {D, -A, x, -u, r, -p, s, -v, y, -B, E, C, -z, w, -t, q, -q, t, -w, z, -C, -E, B, -y, v, -s, p, -r, u, -x, A, -D},   \
        {o, -n, m, -l, k, -j, i, -h, h, -i, j, -k, l, -m, n, -o, -o, n, -m, l, -k, j, -i, h, -h, i, -j, k, -l, m, -n, o},   \
    {                                                                                                                       \
      E, -D, C, -B, A, -z, y, -x, w, -v, u, -t, s, -r, q, -p, p, -q, r, -s, t, -u, v, -w, x, -y, z, -A, B, -C, D, -E        \
    }                                                                                                                       \
  }

constexpr static int16_t g_trCoreDCT2P8[8][8] = DEFINE_DCT2_P8_MATRIX(64, 83, 36, 89, 75, 50, 18);
constexpr static int16_t g_trCoreDCT2P16[16][16] = DEFINE_DCT2_P16_MATRIX(64, 83, 36, 89, 75, 50, 18, 90, 87, 80, 70, 57, 43, 25, 9);
constexpr static int16_t g_trCoreDCT2P32[32][32] = DEFINE_DCT2_P32_MATRIX(64, 83, 36, 89, 75, 50, 18, 90, 87, 80, 70, 57, 43, 25, 9, 90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4);
constexpr static int TRANSFORM_SHIFT = 6;

namespace freq
{
  // fast-fourier transform
  std::vector<std::complex<double>> fft1d(const std::vector<double> &v);
  std::vector<std::complex<double>> fft1d(const std::vector<uint8_t> &v);

  /**
   * @param input NxN size vector
   * @param N Width(or height) of input
   * @param ortho Normalize output result
   * @param inputScale Multiply this value to all of the input values before DFT calculation
   *
   * @return vector<complex<double>> DFT result
   */
  std::vector<std::complex<double>> fft2d_nxn(
      const std::vector<uint8_t> &input,
      size_t N,
      bool ortho = false,
      double inputScale = 1.0);

  // discrete cosine transform type-II
  std::vector<double> dct8x8(const uint8_t *src);
  std::vector<double> dct16x16(const uint8_t *src);
  std::vector<double> dct32x32(const uint8_t *src);

  std::vector<float> dct8x8_simd(const uint8_t *src);

  double coherence_nxn(
      const uint8_t *x,
      const uint8_t *y,
      size_t N,
      const std::string method = "dct2");

  std::vector<double> dct8x8_tc(
      const uint8_t *ref_image,
      int ref_stride,
      const uint8_t *tar_image,
      int tar_stride,
      int width,
      int height);

  std::vector<float> dct8x8_tc_simd(
      const uint8_t *ref_image,
      int ref_stride,
      const uint8_t *tar_image,
      int tar_stride,
      int width,
      int height);

  namespace x86::tx
  {
    static constexpr size_t STEP = 4;

    // Number of elements in a SIMD register
    static constexpr size_t NUM_ELEMENTS = sizeof(__m128i) / sizeof(int);
    // Transform size in units of SIMD register
    static constexpr size_t N(size_t TX_SIZE) { return TX_SIZE / NUM_ELEMENTS; }

    // Reverse order of coefficients in SIMD register
    static inline __m128i reverse32(__m128i x) { return _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)); }

    // Load coefficients into SIMD register
    static inline __m128i loadCoeff(const int *p) { return _mm_loadu_si128((const __m128i *)p); }

    // Load matrix coefficients into SIMD register and widen
    static inline __m128i loadMatrixCoeff(const int16_t *p)
    {
      return _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)p));
    }

    // Store coefficients into SIMD register
    static inline void storeCoeff(const int *p, __m128i c) { _mm_storeu_si128((__m128i *)p, c); }

#if USE_AVX2
    // Load matrix coefficients into SIMD register and widen
    static inline __m256i loadMatrixCoeff2(const int16_t *p)
    {
      return _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)p));
    }

    // Store coefficients into SIMD register
    static inline void storeCoeff(const int *p, __m256i c) { _mm256_storeu_si256((__m256i *)p, c); }
#endif

    template <size_t TX_SIZE>
    static inline void butterfly1(__m128i even[N(TX_SIZE)], __m128i odd[N(TX_SIZE)], const int *src)
    {
      constexpr size_t M = N(TX_SIZE) / 2;

      for (size_t k = 0; k < M; k++)
      {
        const __m128i a = loadCoeff(src + NUM_ELEMENTS * k);
        const __m128i b = reverse32(loadCoeff(src + NUM_ELEMENTS * (2 * M - 1 - k)));
        even[k] = _mm_add_epi32(a, b);
        odd[M + k] = _mm_sub_epi32(a, b);
      }
    }

    template <size_t TX_SIZE, size_t D>
    static inline void butterfly(__m128i even[N(TX_SIZE)], __m128i odd[N(TX_SIZE)])
    {
      constexpr size_t M = N(TX_SIZE) / D;

      if constexpr (M > 0)
      {
        for (size_t k = 0; k < M; k++)
        {
          __m128i a = even[k];
          __m128i b = reverse32(even[2 * M - 1 - k]);
          even[k] = _mm_add_epi32(a, b);
          odd[M + k] = _mm_sub_epi32(a, b);
        }
      }
    }

    template <size_t TX_SIZE, size_t D, bool FIRST>
    static inline void mul(const int16_t m[TX_SIZE][TX_SIZE], const __m128i *src, __m128i dst[TX_SIZE * STEP],
                           size_t numActiveRowsOut, size_t i, size_t numBatchRowsIn)
    {
      constexpr size_t M = N(TX_SIZE) / D;

      if constexpr (M > 0)
      {
#if USE_AVX2
        if constexpr (M % 2 == 0)
        {
          for (size_t k = FIRST ? 0 : D / 2; k < numActiveRowsOut; k += D)
          {
            __m256i sum = _mm256_setzero_si256();

            for (size_t l = 0; l < M / 2; l++)
            {
              __m256i c = loadMatrixCoeff2(&m[k][2 * NUM_ELEMENTS * l]);
              __m256i x = _mm256_loadu_si256((const __m256i *)&src[(FIRST ? 0 : M) + 2 * l]);
              sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(c, x));
            }

            dst[k * numBatchRowsIn + i] = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
          }

          return;
        }
#endif
        for (size_t k = FIRST ? 0 : D / 2; k < numActiveRowsOut; k += D)
        {
          __m128i sum = _mm_setzero_si128();

          for (size_t l = 0; l < M; l++)
          {
            __m128i c = loadMatrixCoeff(&m[k][NUM_ELEMENTS * l]);
            sum = _mm_add_epi32(sum, _mm_mullo_epi32(c, src[(FIRST ? 0 : M) + l]));
          }

          dst[k * numBatchRowsIn + i] = sum;
        }
      }
    }

    template <size_t TX_SIZE>
    static inline void store(const __m128i tmp[TX_SIZE * STEP], int *dst, ptrdiff_t dstStride,
                             size_t numBatchRowsIn, int add, int shift, size_t numActiveRowsOut)
    {
      const int log2numBatchRowsIn = std::min((int)numBatchRowsIn - 1, 2);

      const size_t increment = 4 >> log2numBatchRowsIn;

      for (size_t k = 0; k < numActiveRowsOut << log2numBatchRowsIn >> 2; k++)
      {
        const __m128i *x = tmp + STEP * k;
        const __m128i x01 = _mm_add_epi32(_mm_unpacklo_epi32(x[0], x[1]), _mm_unpackhi_epi32(x[0], x[1]));
        const __m128i x23 = _mm_add_epi32(_mm_unpacklo_epi32(x[2], x[3]), _mm_unpackhi_epi32(x[2], x[3]));
        const __m128i x0123 = _mm_add_epi32(_mm_unpacklo_epi64(x01, x23), _mm_unpackhi_epi64(x01, x23));

        const __m128i y = _mm_sra_epi32(_mm_add_epi32(x0123, _mm_set1_epi32(add)), _mm_cvtsi32_si128(shift));

        storeCoeff(dst + k * increment * dstStride, y);
      }
    }

    template <size_t TX_SIZE>
    static inline void clear(size_t numRowsIn, size_t numActiveRowsIn, size_t numActiveRowsOut,
                             int *dst, ptrdiff_t dstStride)
    {
      if (numRowsIn > numActiveRowsIn)
      {
        for (size_t j = 0; j < numActiveRowsOut; j++)
        {
          for (size_t k = numActiveRowsIn; k < numRowsIn; k += NUM_ELEMENTS)
          {
            storeCoeff(dst + j * dstStride + k, _mm_setzero_si128());
          }
        }
      }

      if (numActiveRowsOut < TX_SIZE)
      {
        for (size_t j = numActiveRowsOut * dstStride; j < TX_SIZE * dstStride; j += NUM_ELEMENTS)
        {
          storeCoeff(dst + j, _mm_setzero_si128());
        }
      }
    }

    template <size_t TX_SIZE, const int16_t M[TX_SIZE][TX_SIZE]>
    static std::vector<int> dct2(const int *src, int shift, int numRowsIn, int numZeroTrailRowsIn, int numZeroTrailRowsOut)
    {
      std::vector<int> dst(TX_SIZE * TX_SIZE);
      const int add = 1 << shift >> 1;
      const size_t numActiveRowsIn = numRowsIn - numZeroTrailRowsIn;
      const ptrdiff_t dstStride = numRowsIn;
      const size_t numActiveRowsOut = TX_SIZE - numZeroTrailRowsOut;

      for (size_t j = 0; j < numActiveRowsIn; j += STEP)
      {
        __m128i tmp[TX_SIZE * STEP];

        const size_t numBatchRowsIn = std::min(numActiveRowsIn - j, STEP);

        for (size_t i = 0; i < numBatchRowsIn; i++)
        {
          static_assert(N(TX_SIZE) > 1); // minimum size of butterfly to apply

          __m128i even[N(TX_SIZE) / 2], odd[N(TX_SIZE)];

          butterfly1<TX_SIZE>(even, odd, src);
          butterfly<TX_SIZE, 4>(even, odd);
          butterfly<TX_SIZE, 8>(even, odd);
          butterfly<TX_SIZE, 16>(even, odd);

          mul<TX_SIZE, N(TX_SIZE), true>(M, even, tmp, numActiveRowsOut, i, numBatchRowsIn);
          mul<TX_SIZE, 16, false>(M, odd, tmp, numActiveRowsOut, i, numBatchRowsIn);
          mul<TX_SIZE, 8, false>(M, odd, tmp, numActiveRowsOut, i, numBatchRowsIn);
          mul<TX_SIZE, 4, false>(M, odd, tmp, numActiveRowsOut, i, numBatchRowsIn);
          mul<TX_SIZE, 2, false>(M, odd, tmp, numActiveRowsOut, i, numBatchRowsIn);

          src += TX_SIZE;
        }

        store<TX_SIZE>(tmp, dst.data() + j, dstStride, numBatchRowsIn, add, shift, numActiveRowsOut);
      }

      clear<TX_SIZE>(numRowsIn, numActiveRowsIn, numActiveRowsOut, dst.data(), dstStride);

      return dst;
    }
  }
}