
project(tc-freq)

add_library(
  ${PROJECT_NAME} 
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/tc.cpp
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
  ${FFTW_LIBRARIES}
  ${FFTW_FLOAT_THREADS_LIB}
)

target_include_directories(${PROJECT_NAME}
  PRIVATE PkgConfig::FFTW
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(${PROJECT_NAME} PRIVATE 
  -Wall 
  -Werror 
  -mavx 
  -mavx2 
  -mssse3 
  -msse4.1 
  -fno-tree-vectorize
  -O3
)