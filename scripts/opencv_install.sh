#!/bin/bash

# Define custom directory for cloning OpenCV source
CLONE_DIR="/mnt/sdb/tmp/opencv"
PREFIX_DIR="/mnt/sdb/storage"

# Detect OS and install necessary packages
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi

case $OS in
    "ubuntu")
        sudo apt-get update
        sudo apt-get install -y git build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev \
            libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
            libxvidcore-dev libx264-dev libgtk-3-dev \
            libatlas-base-dev gfortran
        ;;
    "centos"|"rhel")
        sudo yum update -y
        if [ ${VERSION_ID} -ge 8 ]; then
            sudo dnf install -y epel-release
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y git cmake pkgconfig libjpeg-turbo-devel libtiff-devel jasper-devel libpng-devel \
                libv4l-devel libdc1394-devel \
                ffmpeg-devel gtk3-devel \
                atlas-devel gfortran
        else
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y git cmake pkgconfig libjpeg-turbo-devel libtiff-devel jasper-devel libpng-devel \
                libv4l-devel libdc1394-devel \
                ffmpeg-devel gtk2-devel \
                atlas-devel gfortran
        fi
        ;;
    *)
        echo "Unsupported OS. Exiting."
        exit 1
        ;;
esac

# Clone OpenCV and OpenCV contrib modules
git clone --branch 4.9.0 https://github.com/opencv/opencv.git "$CLONE_DIR"/opencv
git clone --branch 4.9.0 https://github.com/opencv/opencv_contrib.git "$CLONE_DIR"/opencv_contrib

# Create a build directory
mkdir -p "$CLONE_DIR"/opencv/build
cd "$CLONE_DIR"/opencv/build

# Configure OpenCV build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX="$PREFIX_DIR" \
      -D OPENCV_EXTRA_MODULES_PATH="$CLONE_DIR"/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON ..

# Compile and install
make -j$(nproc)
sudo make install

# Verify OpenCV installation
pkg-config --modversion opencv4

# Cleanup: Remove downloaded files after installation
cd ../../../
rm -rf "$CLONE_DIR"
rm -rf "$PREFIX_DIR"
echo "OpenCV installation complete and source files removed."
