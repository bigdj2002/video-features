
project(features)

add_library(
  ${PROJECT_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/glcm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/ncc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/tc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/nlp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/pca.cpp
)

find_package(PkgConfig REQUIRED)

pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
find_library(
  FFTW_FLOAT_THREADS_LIB
  NAMES "fftw3f_threads"
  PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  libmath
  xsimd
  armadillo
  ${FFTW_LIBRARIES}
  ${FFTW_FLOAT_THREADS_LIB}
  ${OpenCV_LIBS}
)

target_include_directories(${PROJECT_NAME}
  PRIVATE PkgConfig::FFTW
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../xsimd/include/xsimd
  ${CMAKE_CURRENT_SOURCE_DIR}/../armadillo/include/
)

target_compile_options(${PROJECT_NAME} PRIVATE 
  -Wall 
  -Werror 
  -mavx 
  -mavx2 
  -mssse3 
  -msse4.1 
  -fno-tree-vectorize
  # -O3
)