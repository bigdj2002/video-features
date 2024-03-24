#include <type_traits>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstring>
#include <iostream>

#include "tc.h"
#include "mathUtil.h"

template <typename Ty1, typename Ty2>
std::vector<std::complex<Ty1>> fft1d_impl(const std::vector<Ty2> &v)
{
  static_assert(std::is_same<Ty1, float>::value || std::is_same<Ty1, double>::value,
                "Template argument must be float or double");

  using fft_complext_t = typename std::conditional<std::is_same<Ty1, double>::value, fftw_complex, fftwf_complex>::type;

  const size_t N = v.size();
  fft_complext_t *fft_in = nullptr;
  fft_complext_t *fft_out = nullptr;
  std::vector<std::complex<Ty1>> fft_result(N);

  if (std::is_same<Ty1, double>::value)
  {
    fft_in = (fft_complext_t *)fftw_malloc(sizeof(fft_complext_t) * N);
    fft_out = (fft_complext_t *)fftw_malloc(sizeof(fft_complext_t) * N);

    for (size_t i = 0; i < N; ++i)
    {
      fft_in[i][0] = static_cast<Ty1>(v[i]);
      fft_in[i][1] = 0;
    }

    fftw_plan plan = fftw_plan_dft_1d(N, (fftw_complex *)fft_in, (fftw_complex *)fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (size_t i = 0; i < N; ++i)
    {
      fft_result[i] = {fft_out[i][0], fft_out[i][1]};
    }

    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
  }
  else
  {
    fft_in = (fft_complext_t *)fftwf_malloc(sizeof(fft_complext_t) * N);
    fft_out = (fft_complext_t *)fftwf_malloc(sizeof(fft_complext_t) * N);

    for (int i = 0; i < N; ++i)
    {
      fft_in[i][0] = static_cast<Ty1>(v[i]);
      fft_in[i][1] = 0;
    }

    fftwf_plan plan = fftwf_plan_dft_1d(N, (fftwf_complex *)fft_in, (fftwf_complex *)fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);

    for (int i = 0; i < N; ++i)
    {
      fft_result[i] = {fft_out[i][0], fft_out[i][1]};
    }

    fftwf_destroy_plan(plan);
    fftwf_free(fft_in);
    fftwf_free(fft_out);
  }

  return fft_result;
}

template <typename Ty1, typename Ty2>
std::vector<std::complex<Ty1>> fft2d_nxn_impl(
    const std::vector<Ty2> &src,
    size_t N)
{
  static_assert(std::is_same<Ty1, float>::value || std::is_same<Ty1, double>::value,
                "Template argument must be float or double");

  if (src.size() != N * N)
    throw std::invalid_argument("src.size() must be equal to N^2");

  if (!math_util::IsPowerOfTwo(N) || N < 2u)
    throw std::invalid_argument("N must be a power of two integer numbers such as 2, 4, 8, ...");

  using fft_complext_t = typename std::conditional<std::is_same<Ty1, double>::value, fftw_complex, fftwf_complex>::type;

  fft_complext_t *data = nullptr;
  std::vector<std::complex<Ty1>> result(N * N);

  if (std::is_same<Ty1, double>::value)
  {
    data = (fft_complext_t *)fftw_malloc(sizeof(fft_complext_t) * N * N);

    for (size_t i = 0; i < N * N; ++i)
    {
      data[i][0] = static_cast<Ty1>(src[i]);
      data[i][1] = 0;
    }

    fftw_plan plan = fftw_plan_dft_2d(N, N, (fftw_complex *)data, (fftw_complex *)data, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for (size_t i = 0; i < N * N; ++i)
    {
      result[i] = {data[i][0], data[i][1]};
    }

    fftw_free(data);
  }
  else
  {
    data = (fft_complext_t *)fftwf_malloc(sizeof(fft_complext_t) * N * N);

    for (int i = 0; i < N * N; ++i)
    {
      data[i][0] = static_cast<Ty1>(src[i]);
      data[i][1] = 0;
    }

    fftwf_plan plan = fftwf_plan_dft_2d(N, N, (fftwf_complex *)data, (fftwf_complex *)data, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    for (int i = 0; i < N * N; ++i)
    {
      result[i] = {data[i][0], data[i][1]};
    }

    fftwf_free(data);
  }

  return result;
}

// template<typename Ty1, typename Ty2>
// std::vector<std::complex<Ty1>> calc_cross_spectral_density(
//   const std::vector<Ty2>& v1,
//   const std::vector<Ty2>& v2)
// {
//   static_assert(std::is_same<Ty1, float>::value || std::is_same<Ty1, double>::value,
//                 "Template argument must be float or double");

//   if(v1.size() != v2.size())
//     throw invalid_argument("v1.size() and v2.size() must be equal");

//   const size_t N = v1.size();

//   auto fft1 = fft1d_impl<Ty1, Ty2>(v1);
//   auto fft2 = fft1d_impl<Ty1, Ty2>(v2);

//   std::vector<std::complex<Ty1>> csd(N);
//   for(size_t i=0; i<N; ++i) {
//     csd[i] = fft1[i] * std::conj(fft2[i]);
//   }

//   return csd;
// }

static void fastForwardDCT2_8x8(
    const int *src,
    int *dst,
    int shift)
{
  int E[4], O[4];
  int EE[2], EO[2];
  int add = (shift > 0) ? (1 << (shift - 1)) : 0;
  const int16_t *iT = g_trCoreDCT2P8[0];

  __attribute__((unused)) int *pCoef = dst;
  for (int j = 0; j < 8; ++j)
  {
    for (int k = 0; k < 4; ++k)
    {
      E[k] = src[k] + src[7 - k];
      O[k] = src[k] - src[7 - k];
    }

    EE[0] = E[0] + E[3];
    EO[0] = E[0] - E[3];
    EE[1] = E[1] + E[2];
    EO[1] = E[1] - E[2];

    dst[0] = (iT[0] * EE[0] + iT[1] * EE[1] + add) >> shift;
    dst[4 * 8] = (iT[32] * EE[0] + iT[33] * EE[1] + add) >> shift;
    dst[2 * 8] = (iT[16] * EO[0] + iT[17] * EO[1] + add) >> shift;
    dst[6 * 8] = (iT[48] * EO[0] + iT[49] * EO[1] + add) >> shift;

    dst[8] = (iT[8] * O[0] + iT[9] * O[1] + iT[10] * O[2] + iT[11] * O[3] + add) >> shift;
    dst[3 * 8] = (iT[24] * O[0] + iT[25] * O[1] + iT[26] * O[2] + iT[27] * O[3] + add) >> shift;
    dst[5 * 8] = (iT[40] * O[0] + iT[41] * O[1] + iT[42] * O[2] + iT[43] * O[3] + add) >> shift;
    dst[7 * 8] = (iT[56] * O[0] + iT[57] * O[1] + iT[58] * O[2] + iT[59] * O[3] + add) >> shift;

    src += 8;
    dst++;
  }
}

static void fastForwardDCT2_16x16(
    const int *src,
    int *dst,
    int shift)
{
  int E[8], O[8];
  int EE[4], EO[4];
  int EEE[2], EEO[2];
  int add = (shift > 0) ? (1 << (shift - 1)) : 0;

  const int16_t *iT = g_trCoreDCT2P16[0];
  __attribute__((unused)) int *pCoef = dst;

  for (int j = 0; j < 16; ++j)
  {
    for (int k = 0; k < 8; ++k)
    {
      E[k] = src[k] + src[15 - k];
      O[k] = src[k] - src[15 - k];
    }
    for (int k = 0; k < 4; ++k)
    {
      EE[k] = E[k] + E[7 - k];
      EO[k] = E[k] - E[7 - k];
    }

    EEE[0] = EE[0] + EE[3];
    EEO[0] = EE[0] - EE[3];
    EEE[1] = EE[1] + EE[2];
    EEO[1] = EE[1] - EE[2];

    dst[0] = (iT[0] * EEE[0] + iT[1] * EEE[1] + add) >> shift;
    dst[8 * 16] = (iT[8 * 16] * EEE[0] + iT[8 * 16 + 1] * EEE[1] + add) >> shift;
    dst[4 * 16] = (iT[4 * 16] * EEO[0] + iT[4 * 16 + 1] * EEO[1] + add) >> shift;
    dst[12 * 16] = (iT[12 * 16] * EEO[0] + iT[12 * 16 + 1] * EEO[1] + add) >> shift;

    for (int k = 2; k < 16; k += 4)
    {
      dst[k * 16] = (iT[k * 16] * EO[0] + iT[k * 16 + 1] * EO[1] + iT[k * 16 + 2] * EO[2] + iT[k * 16 + 3] * EO[3] + add) >> shift;
    }

    for (int k = 1; k < 16; k += 2)
    {
      dst[k * 16] = (iT[k * 16] * O[0] + iT[k * 16 + 1] * O[1] + iT[k * 16 + 2] * O[2] + iT[k * 16 + 3] * O[3] +
                     iT[k * 16 + 4] * O[4] + iT[k * 16 + 5] * O[5] + iT[k * 16 + 6] * O[6] + iT[k * 16 + 7] * O[7] + add) >>
                    shift;
    }

    src += 16;
    dst++;
  }
}

static void fastForwardDCT2_32x32(
    const int *src,
    int *dst,
    int shift)
{
  int E[16], O[16];
  int EE[8], EO[8];
  int EEE[4], EEO[4];
  int EEEE[2], EEEO[2];
  int add = (shift > 0) ? (1 << (shift - 1)) : 0;

  const int16_t *iT = g_trCoreDCT2P32[0];

  __attribute__((unused)) int *pCoef = dst;
  for (int j = 0; j < 32; ++j)
  {
    for (int k = 0; k < 16; ++k)
    {
      E[k] = src[k] + src[31 - k];
      O[k] = src[k] - src[31 - k];
    }
    for (int k = 0; k < 8; ++k)
    {
      EE[k] = E[k] + E[15 - k];
      EO[k] = E[k] - E[15 - k];
    }
    for (int k = 0; k < 4; ++k)
    {
      EEE[k] = EE[k] + EE[7 - k];
      EEO[k] = EE[k] - EE[7 - k];
    }
    /* EEEE and EEEO */
    EEEE[0] = EEE[0] + EEE[3];
    EEEO[0] = EEE[0] - EEE[3];
    EEEE[1] = EEE[1] + EEE[2];
    EEEO[1] = EEE[1] - EEE[2];

    dst[0] = (iT[0 * 32 + 0] * EEEE[0] + iT[0 * 32 + 1] * EEEE[1] + add) >> shift;
    dst[16 * 32] = (iT[16 * 32 + 0] * EEEE[0] + iT[16 * 32 + 1] * EEEE[1] + add) >> shift;
    dst[8 * 32] = (iT[8 * 32 + 0] * EEEO[0] + iT[8 * 32 + 1] * EEEO[1] + add) >> shift;
    dst[24 * 32] = (iT[24 * 32 + 0] * EEEO[0] + iT[24 * 32 + 1] * EEEO[1] + add) >> shift;
    for (int k = 4; k < 32; k += 8)
    {
      dst[k * 32] = (iT[k * 32 + 0] * EEO[0] + iT[k * 32 + 1] * EEO[1] + iT[k * 32 + 2] * EEO[2] + iT[k * 32 + 3] * EEO[3] + add) >> shift;
    }
    for (int k = 2; k < 32; k += 4)
    {
      dst[k * 32] = (iT[k * 32 + 0] * EO[0] + iT[k * 32 + 1] * EO[1] + iT[k * 32 + 2] * EO[2] + iT[k * 32 + 3] * EO[3] +
                     iT[k * 32 + 4] * EO[4] + iT[k * 32 + 5] * EO[5] + iT[k * 32 + 6] * EO[6] + iT[k * 32 + 7] * EO[7] + add) >>
                    shift;
    }
    for (int k = 1; k < 32; k += 2)
    {
      dst[k * 32] = (iT[k * 32 + 0] * O[0] + iT[k * 32 + 1] * O[1] + iT[k * 32 + 2] * O[2] + iT[k * 32 + 3] * O[3] +
                     iT[k * 32 + 4] * O[4] + iT[k * 32 + 5] * O[5] + iT[k * 32 + 6] * O[6] + iT[k * 32 + 7] * O[7] +
                     iT[k * 32 + 8] * O[8] + iT[k * 32 + 9] * O[9] + iT[k * 32 + 10] * O[10] + iT[k * 32 + 11] * O[11] +
                     iT[k * 32 + 12] * O[12] + iT[k * 32 + 13] * O[13] + iT[k * 32 + 14] * O[14] + iT[k * 32 + 15] * O[15] + add) >>
                    shift;
    }
    src += 32;
    dst++;
  }
}

// template<typename T>
// T frequencyCorrelation(
//   const std::vector<std::complex<T>>& x,
//   const std::vector<std::complex<T>>& y)
// {
//   if(x.size() != y.size() || x.empty())
//     throw std::invalid_argument("x.size() must be eqaul to y.size() and both of x and y must have at least one value");

//   const size_t N = x.size();
//   T numer = 0;
//   T denom_x = 0;
//   T denom_y = 0;

//   for(size_t i=0; i< N; ++i) {
//     complex<T> d = x[i] * conj(y[i]);
//     numer += (d.real() * d.real() + d.imag() * d.imag());
//     denom_x += (x[i] * conj(x[i])).real();
//     denom_y += (y[i] * conj(y[i])).real();
//   }

//   T result = 0;

//   if(abs(denom_x + denom_y) < std::numeric_limits<T>::epsilon())
//     result = T(1.0);
//   else
//     result = numer / (std::sqrt(denom_x) * std::sqrt(denom_y));

//   return result;
// }

namespace freq
{
  std::vector<std::complex<double>> fft1d(const std::vector<double> &v)
  {
    return fft1d_impl<double, double>(v);
  }

  std::vector<std::complex<double>> fft1d(const std::vector<uint8_t> &v)
  {
    return fft1d_impl<double, uint8_t>(v);
  }

  std::vector<std::complex<double>> fft2d_nxn(
      const std::vector<uint8_t> &input,
      size_t N,
      bool ortho,
      double inputScale)
  {
    const double scale = inputScale * (ortho ? (1.0 / N) : 1.0);

    if (std::abs(scale - 1.0) < std::numeric_limits<double>::epsilon())
    {
      return fft2d_nxn_impl<double, uint8_t>(input, N);
    }
    else
    {
      std::vector<double> dinput(input.size());

      for (size_t i = 0; i < input.size(); ++i)
      {
        dinput[i] = input[i] * scale;
      }

      return fft2d_nxn_impl<double, double>(dinput, N);
    }
  }

  std::vector<double> dct8x8(const uint8_t *src)
  {
    int buffer1[64], buffer2[64];
    std::vector<double> result(64u);

    for (int i = 0; i < 64; ++i)
      buffer1[i] = static_cast<int>(src[i]);

    fastForwardDCT2_8x8(buffer1, buffer2, 0);
    fastForwardDCT2_8x8(buffer2, buffer1, 0);

    int scale_bits =
        3 +                  // log2(8)
        8 +                  // bit_depth
        2 * TRANSFORM_SHIFT; // 2 * transform shift

    double scale = 1.0 / (1 << scale_bits);

    for (int i = 0; i < 64; ++i)
      result[i] = buffer1[i] * scale;

    return result;
  }

  std::vector<float> dct8x8_simd(const uint8_t *src)
  {
    int buffer1[64], buffer2[64];
    std::vector<float> result(64u);

    for (int i = 0; i < 64; ++i)
      buffer1[i] = static_cast<int>(src[i]);

    fastForwardDCT2_8x8(buffer1, buffer2, 0);
    fastForwardDCT2_8x8(buffer2, buffer1, 0);

    int scale_bits =
        3 +                  // log2(8)
        8 +                  // bit_depth
        2 * TRANSFORM_SHIFT; // 2 * transform shift

    float scale = 1.0 / (1 << scale_bits);

    using batch_type = xsimd::batch<float>;
    const size_t batch_size = batch_type::size;
    const size_t algined_size = 64 - 64 % batch_size;

    for (size_t i = 0; i < algined_size; i += batch_size)
    {
      auto result_batch = xsimd::mul(xsimd::batch_cast<float>(xsimd::load_unaligned(&buffer1[i])), batch_type(scale));
      xsimd::store_unaligned(&result[i], result_batch);
    }

    for (int i = algined_size; i < 64; ++i)
      result[i] = buffer1[i] * scale;

    return result;
  }

  std::vector<double> dct16x16(const uint8_t *src)
  {
    int buffer1[256], buffer2[256];
    std::vector<double> result(256u);

    for (int i = 0; i < 256; ++i)
      buffer1[i] = static_cast<int>(src[i]);

    fastForwardDCT2_16x16(buffer1, buffer2, 1);
    fastForwardDCT2_16x16(buffer2, buffer1, 1);

    int scale_bits =
        4                     // log2(16)
        + 8                   // bit_depth
        + 2 * TRANSFORM_SHIFT // 2 * transform shift
        - 2;                  // two of right shift 1 from fastForwardDCT2_16x16

    double scale = 1.0 / (1 << scale_bits);

    for (int i = 0; i < 256; ++i)
      result[i] = buffer1[i] * scale;

    return result;
  }

  std::vector<double> dct32x32(const uint8_t *src)
  {
    int buffer1[1024], buffer2[1024];
    std::vector<double> result(1024u);

    for (int i = 0; i < 1024; ++i)
      buffer1[i] = static_cast<int>(src[i]);

    fastForwardDCT2_32x32(buffer1, buffer2, 2);
    fastForwardDCT2_32x32(buffer2, buffer1, 2);

    int scale_bits =
        5                     // log2(32)
        + 8                   // bit_depth
        + 2 * TRANSFORM_SHIFT // 2 * transform shift
        - 4;                  // two of right shift 1 from fastForwardDCT2_32x32

    double scale = 1.0 / (1 << scale_bits);

    for (int i = 0; i < 1024; ++i)
      result[i] = buffer1[i] * scale;

    return result;
  }

  double coherence_nxn(
      const uint8_t *x,
      const uint8_t *y,
      size_t N,
      const std::string method)
  {
    (void)method;

    std::vector<double> X, Y;

    switch (N)
    {
    case 8:
      X = dct8x8(x);
      Y = dct8x8(y);
      break;
    case 16:
      X = dct16x16(x);
      Y = dct16x16(y);
      break;
    case 32:
      X = dct32x32(x);
      Y = dct32x32(y);
      break;
    default:
      throw std::invalid_argument("Unsupported N size. N must be one of {8, 16, 32}");
      return 0;
    }

    return math_util::PearsonCorrelationCoefficients(X, Y);
  }

  std::vector<double> dct8x8_tc(
      const uint8_t *ref_image,
      int ref_stride,
      const uint8_t *tar_image,
      int tar_stride,
      int width,
      int height)
  {
    int nblkx = width / 8;
    int nblky = height / 8;
    uint8_t block_cache1[64], block_cache2[64];

    std::vector<double> r;
    r.reserve(nblkx * nblky);

    for (int by = 0; by < nblky; ++by)
    {
      for (int bx = 0; bx < nblkx; ++bx)
      {
        __attribute__((unused)) int ix = bx * 8;
        __attribute__((unused)) int iy = by * 8;

        for (int k = 0; k < 8; ++k)
        {
          std::memcpy(&block_cache1[8 * k], &ref_image[(k + iy) * ref_stride + ix], 8);
          std::memcpy(&block_cache2[8 * k], &tar_image[(k + iy) * tar_stride + ix], 8);
        }
        auto c1 = dct8x8(block_cache1);
        auto c2 = dct8x8(block_cache2);

        r.emplace_back(math_util::PearsonCorrelationCoefficients(c1, c2));
      }
    }

    return r;
  }

  std::vector<float> dct8x8_tc_simd(
      const uint8_t *ref_image,
      int ref_stride,
      const uint8_t *tar_image,
      int tar_stride,
      int width,
      int height)
  {
    int nblkx = width / 8;
    int nblky = height / 8;
    int block_cache1[64], block_cache2[64];

    std::vector<float> r;
    r.reserve(nblkx * nblky);

    for (int by = 0; by < nblky; ++by)
    {
      for (int bx = 0; bx < nblkx; ++bx)
      {
        __attribute__((unused)) int ix = bx * 8;
        __attribute__((unused)) int iy = by * 8;

        for (int k = 0; k < 8; ++k)
        {
          for (int l = 0; l < 8; ++l)
          {
            block_cache1[8 * k + l] = static_cast<int32_t>(ref_image[(k + iy) * 8 + ix + l]);
            block_cache2[8 * k + l] = static_cast<int32_t>(tar_image[(k + iy) * 8 + ix + l]);
          }
        }

        int scale_bits = 3 /* log2(8) */ + 8 /* bit_depth */ + TRANSFORM_SHIFT /* transform shift */ - 5 /* ??? */;
        float scale = 1.0 / (1 << scale_bits);
        using batch_type = xsimd::batch<float>;
        const size_t batch_size = batch_type::size;
        const size_t algined_size = 64 - 64 % batch_size;

        const int shift_1st = (std::log2(8) + 8 /* bitDepth */ + TRANSFORM_SHIFT /* transform shift */) - 15 /* maxLog2TrDynamicRange */;
        const int shift_2nd = std::log2(8) + TRANSFORM_SHIFT /* transform shift */;
        auto row_t1 = x86::tx::dct2<8, g_trCoreDCT2P8>(block_cache1, shift_1st, 8, 0, 0);
        auto full_t1 = x86::tx::dct2<8, g_trCoreDCT2P8>(row_t1.data(), shift_2nd, 8, 0, 0);
        std::vector<float> c1_simd(64);
        for (size_t i = 0; i < algined_size; i += batch_size)
        {
          auto result_batch = xsimd::mul(xsimd::batch_cast<float>(xsimd::load_unaligned(&full_t1[i])), batch_type(scale));
          xsimd::store_unaligned(&c1_simd[i], result_batch);
        }

        auto row_t2 = x86::tx::dct2<8, g_trCoreDCT2P8>(block_cache2, shift_1st, 8, 0, 0);
        auto full_t2 = x86::tx::dct2<8, g_trCoreDCT2P8>(row_t2.data(), shift_2nd, 8, 0, 0);
        std::vector<float> c2_simd(64);
        for (size_t i = 0; i < algined_size; i += batch_size)
        {
          auto result_batch = xsimd::mul(xsimd::batch_cast<float>(xsimd::load_unaligned(&full_t2[i])), batch_type(scale));
          xsimd::store_unaligned(&c2_simd[i], result_batch);
        }

        r.emplace_back(math_util::PearsonCorrelationCoefficients_simd(c1_simd, c2_simd));
      }
    }

    return r;
  }
}
