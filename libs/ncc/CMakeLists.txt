
project(ncc)

add_library(
  ${PROJECT_NAME} 
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/ncc.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  libmath
  xsimd
)

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../xsimd/include/xsimd
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