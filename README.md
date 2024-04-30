# Video Features Generator

The Video Features Generator is a C++ application designed for extracting and analyzing various video features from YUV video files. It leverages modern computational methods to process video frames efficiently, calculating features essential for tasks in video analysis and processing.

## Features

- **File Support**: Only YUV video file formats are supported.
## Feature Calculations

- **This application extracts several key features from video data, each providing unique insights into the video's characteristics:**
  - **Gray Level Co-occurrence Matrix (GLCM)**: A statistical method that analyzes the texture of an image by assessing how often pairs of pixel with specific values and spatial relationships occur in an image, useful for understanding textural features like contrast, homogeneity, and entropy which contribute to the perception of image quality.
  - **Normalized Cross-Correlation (NCC)**: Measures the similarity between two images or templates, especially robust for video as it compensates for changes in brightness. NCC is crucial for motion estimation and tracking, making it ideal for comparing different frames in a video sequence.
  - **Temporal Coherence (TC)**: Evaluates the consistency or smoothness of motion between successive frames by assessing the spectral coherence of their frequency components. Higher temporal coherence indicates less variation between frames, suggesting smoother motion, and is vital for effective video compression and prediction models.
  - **Perceptual Quality Assessment (NLP - Normalized Laplacian Pyramid)**: Contrary to the traditional use of NLP for text, in this context, it refers to a perceptual image quality metric. The application uses a Normalized Laplacian Pyramid approach to assess video quality, which mimics early visual system processing. It involves subtracting a blurred version of the image (local mean subtraction) and normalizing by an estimate of local amplitude, effectively reducing image redundancy and enhancing perceptual quality evaluation.
  - **Principal Component Analysis (PCA)**: Used to reduce the dimensionality of the dataset by transforming it into a set of values of linearly uncorrelated variables called principal components. This helps in simplifying the data while preserving as much variability as possible.

- **Each of these features contributes to a robust analysis of video quality, texture, and motion, facilitating advanced video processing tasks.**

- **Performance**:
  - Multithreaded processing to leverage multi-core processors for enhanced performance.
  - SIMD (Single Instruction, Multiple Data) optimizations for data parallelism.
- **Output**: Exports results in JSON format for easy integration with data analysis tools and pipelines.

## Prerequisites

Ensure these prerequisites are installed before compiling and running the application:
- C++17 compatible compiler (GCC or Clang recommended)
- CMake (version 3.12 or higher)
- OpenCV (for image processing tasks)
- Armadillo (for linear algebra functions)
- FFTW3 (for Fourier transform operations)

## Setup

To set up the project, clone the repository and build the application:

```bash
git clone https://github.com/your-repository/video-features.git
cd video-features
mkdir build && cd build
cmake ..
make
```

## Usage

Run the application with the following command from the build directory:

```bash
./feature_generator -i path_to_yuv_file -o output_json_path [options]
```

### Command-line Options

- `-i <path>`: Path to the input YUV file.
- `-o <path>`: Path to the output JSON file where extracted features will be saved.
- `-w <width>`: Width of the input video (default is 1920).
- `-h <height>`: Height of the input video (default is 1080).
- `-f <fps>`: Frames per second of the input video (default is 30).
- `-k <seconds>`: Keyframe interval in seconds (default is 1.0).
- `-bs <size>`: Size of the blocks processed (default is 32).
- `-p <count>`: Number of principal components to output after PCA (default is 1).
- `-s <flag>`: Enable SIMD optimizations (0 for off, 1 for on; default is 0).
- `-t <threads>`: Number of threads to use for processing (default is 40).

## Contributing

Contributions to the Video Features Generator are welcome! Please fork the repository and submit pull requests with your suggested features or fixes.