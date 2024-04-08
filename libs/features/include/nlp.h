//
//  Originally adapted and modified from the project "laplacian-pyramid" by Luca Ritz.
//  GitHub project link: https://github.com/LucaRitz/laplacian-pyramid/tree/main
//  This codes (laplacian namespace in nlp.h, nlp.cpp) have been modified to fit the specific requirements and functionalities of the current project.
//

#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>

#include <opencv4/opencv2/opencv.hpp>

#if defined(_MSC_VER)
// Windows
#if defined(LAPLACIAN_PYRAMID_IMPORT)
#define EXPORT_LAPLACIAN_PYRAMID __declspec(dllimport)
#else
#define EXPORT_LAPLACIAN_PYRAMID __declspec(dllexport)
#endif
#elif defined(__GNUC__)
//  GCC
#define EXPORT_LAPLACIAN_PYRAMID __attribute__((visibility("default")))
#else
//  do nothing and hope for the best?
#define EXPORT_LAPLACIAN_PYRAMID
#pragma warning Unknown dynamic link import / export semantics.
#endif

namespace nlp
{
    extern std::vector<double> nlp(
        const uint8_t *ref_image,
        uint32_t ref_stride,
        const uint8_t *tar_image,
        uint32_t tar_stride,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t block_size,
        int ddof = 1);

    extern std::vector<float> nlp_simd(
        const uint8_t *ref_image,
        uint32_t ref_stride,
        const uint8_t *tar_image,
        uint32_t tar_stride,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t block_size,
        int ddof = 1);
}

namespace laplacian
{
    const uint8_t DEFAULT_COMPRESSIONS = 5;
    const float DEFAULT_QUANTIZATION = 0.0f;
    const float DEFAULT_A = 1.0f;

    class EXPORT_LAPLACIAN_PYRAMID LaplacianPyramidException : public std::runtime_error
    {
    public:
        explicit LaplacianPyramidException(const std::string &message = "");
    };

    class EXPORT_LAPLACIAN_PYRAMID LaplacianPyramid
    {
    public:
        /**
         *
         * Creates a laplacian pyramid for the given image with the expected compression levels and quantization.
         * The image has to be single channeled and CV_32F encoded.
         * The default quantization is zero, which means no quantization is applied and the full laplacian planes
         * are being used.
         *
         * @param image The image to encode.
         * @param compressions The compression levels.
         * @param quantization The quantization used for the reduction of entropy.
         */
        explicit LaplacianPyramid(const cv::Mat &image,
                                  uint8_t compressions = DEFAULT_COMPRESSIONS,
                                  float quantization = DEFAULT_QUANTIZATION);

        /**
         *
         * Decodes the pyramid into the original image.
         *
         * @return The image resulting from the decoding process.
         *
         */
        [[nodiscard]] cv::Mat decode() const;

        /**
         *
         * Gets an encoded laplacian image at the expected level.
         *
         * @param level The compression level of the expected laplacian image.
         *
         * @return The laplacian image at the given level.
         */
        [[nodiscard]] cv::Mat at(uint8_t level) const;

        /**
         *
         * Gets a reference to the encoded image at the expected level.
         *
         * @param level The compression level of the expected laplacian image.
         *
         * @return A reference to the laplacian image at the given level.
         */
        [[nodiscard]] cv::Mat &at(uint8_t level);

        /**
         *
         * Gets an encoded laplacian image at the expected level.
         *
         * @param level The compression level of the expected laplacian image.
         *
         * @return The laplacian image at the given level.
         */
        [[nodiscard]] cv::Mat operator[](uint8_t level) const;

        /**
         *
         * Gets a reference to the encoded image at the expected level.
         *
         * @param level The compression level of the expected laplacian image.
         *
         * @return A reference to the laplacian image at the given level.
         */
        [[nodiscard]] cv::Mat &operator[](uint8_t level);

        /**
         *
         * Gets the levels of the pyramid.
         *
         * @return The levels of the pyramid.
         */
        [[nodiscard]] uint8_t levels() const;

        /**
         *
         * Gets the entire laplacian pyramid as a vector of images.
         * Each element in the vector corresponds to a level in the pyramid,
         * with the first element being the base level (lowest resolution)
         * and the last element being the original image or the highest resolution level.
         *
         * This method allows for accessing the entire set of laplacian images
         * that were generated during the pyramid creation process. It is useful
         * for operations that require processing or analyzing multiple levels
         * of the pyramid simultaneously.
         *
         * @return A vector containing all levels of the laplacian pyramid as cv::Mat objects.
         */
        [[nodiscard]] std::vector<cv::Mat> getPyramid() const;

    private:
        std::vector<cv::Mat> _laplacianPlanesQuantized;
        cv::Mat _kernel;

        /**
         *
         * Validates the given image and applies a valid scaling to perform the fast formulas.
         * The resulting image shares the same memory with the given image.
         * If the image cannot be scaled down by the expected compressions, a #laplacian::LaplacianPyramidException is thrown.
         *
         * @param image The image to validate and scale.
         * @param compressions The compressions or levels of the pyramid.
         *
         * @return The scaled image.
         */
        [[nodiscard]] cv::Mat applyValidScaling(const cv::Mat &image, uint8_t compressions) const;

        /**
         *
         * Removes the last column of the given image.
         * The resulting image shares the same memory with the original image.
         *
         * @param image The original image.
         *
         * @return The image without the last column.
         */
        [[nodiscard]] cv::Mat removeLastColumn(const cv::Mat &image) const;

        /**
         *
         * Removes the last row of the given image.
         * The resulting image shares the same memory with the original image.
         *
         * @param image The original image.
         *
         * @return The image without the last row.
         */
        [[nodiscard]] cv::Mat removeLastRow(const cv::Mat &image) const;

        /**
         *
         * Cuts the given image to the size of the given rows and columns starting on the left upper corner (0, 0).
         *
         * @param image The image which is to cut.
         * @param rows The expected rows of the cut image.
         * @param cols The expected columns of the cut image.
         *
         * @return The cut image sharing the same memory with the original image.
         *
         */
        [[nodiscard]] cv::Mat cutImage(const cv::Mat &image, int rows, int cols) const;

        /**
         *
         * Checks if a valid scaling is applied.
         * The formula is taken from the paper "The Laplacian Pyramid as a Compact Image Code". The presented formula
         * may be wrong, for the implementation the followin is used: M_c = (C + 1) / 2^N.
         * Because the matrices in the paper start with an index of -2, the corresponding part of the formula is moved
         * by +2. This results in the final formula: M_c = (C + 3) / 2^N.
         *
         * The dimension is valid when the above formula results in an integer value for M_c.
         *
         * @param dimension The dimension "C" which is to be checked to validity
         * @param compressions The levels of the pyramid. This refers to "N" in the above formula.
         *
         * @return Gives true, when the formula gets an integer value for M_c
         */
        [[nodiscard]] bool isValidScaling(int dimension, uint8_t compressions) const;

        /**
         *
         * Checks if a given value is an integer type. It gets true, when the integer is nearly int.
         *
         * @param value The value to check.
         * @return Gets true, if the value is of type int or nearly of type int.
         */
        [[nodiscard]] bool isInteger(float value) const;

        /**
         *
         * Checks if two types are nearly the same.
         *
         * @param value1 The first value
         * @param value2 The last value
         * @return Gets true, when the first and the last value are nearly the same.
         */
        [[nodiscard]] bool isNearlyEqual(float value1, float value2) const;

        /**
         *
         * Gets the default kernel "w" presented in the paper "The Laplacian Pyramid as a Compact Image Code".
         * The kernel has the following properties: w(2) = a, w(0) = w(4) = 1/4 - a/2, w(1) = w(3) = 1/4
         * Note that the indexes are moved by +2.
         *
         * This one dimensional kernel "w" is transposed and multiplied by itself to generate the 2D kernel matrix.
         *
         * @param a A value to modify the kernel
         * @return The 2D kernel described above
         */
        [[nodiscard]] cv::Mat kernel(float a = DEFAULT_A) const;

        /**
         *
         * Reduces the given image with the given kernel by the given compressions and gets the results in a vector.
         *
         * @param image The image to encode
         * @param kernel The kernel for encoding
         * @param compressions The expected compression levels.
         * @return A vector of the reduced images.
         */
        [[nodiscard]] std::vector<cv::Mat> reduceToGaussians(const cv::Mat &image,
                                                             const cv::Mat &kernel,
                                                             uint8_t compressions) const;

        /**
         *
         * Reduces an image with the given kernel to the expected size (rows, columns).
         *
         * @param image The image to reduce.
         * @param kernel The kernel used for reduction.
         * @param rows The expected rows of the reduced image.
         * @param columns The expected columns of the reduced image.
         * @return The reduced image.
         */
        [[nodiscard]] cv::Mat reduceGaussian(const cv::Mat &image,
                                             const cv::Mat &kernel,
                                             int rows,
                                             int columns) const;

        /**
         *
         * Upsamples the given images to the double of their sizes.
         *
         * @param images The images which are to be upsampled.
         * @param kernel The kernel used for upsampling.
         *
         * @return The upsampled images of a corresponding level.
         */
        [[nodiscard]] std::vector<cv::Mat> upsample(const std::vector<cv::Mat> &images,
                                                    const cv::Mat &kernel) const;

        /**
         *
         * Upsamples the given image to the given row and column size.
         *
         * @param image The image which is to be upsampled.
         * @param rows The expected row size
         * @param cols The expeced column size
         * @param kernel The kernel used for upsampling.
         *
         * @return The upsampled image.
         */
        [[nodiscard]] cv::Mat upsample(const cv::Mat &image, int rows, int cols, const cv::Mat &kernel) const;

        /**
         *
         * Creates the laplacian planes from the given gaussians and upsampled images.
         * The laplacian image of level "n" is the level "n" image of the gaussians.
         * The both vectors have to be of the same length.
         *
         * @param gaussians The gaussian images.
         * @param upsampled The upsampled images.
         *
         * @return The laplacian images, which are basically the difference images of the given vectors.
         */
        [[nodiscard]] std::vector<cv::Mat> buildLaplacianPlanes(const std::vector<cv::Mat> &gaussians,
                                                                const std::vector<cv::Mat> &upsampled);

        /**
         *
         * Quantize uniformly the laplacian planes with the given quantization. The quantization is the delta
         * of one quantization bin. All the values inside a bin are represented by its middle value.
         * The amount of bins is getting smaller in lower levels of the pyramid.
         *
         * @param laplacianPlanes The laplacian planes which are to be quantized.
         * @param quantization The uniform quantization of the laplacian planes.
         *
         * @return The quantized laplacian planes
         */
        [[nodiscard]] std::vector<cv::Mat> quantize(const std::vector<cv::Mat> &laplacianPlanes,
                                                    float quantization) const;
    };
}