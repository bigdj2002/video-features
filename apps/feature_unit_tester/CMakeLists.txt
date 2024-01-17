project(feature_unit_tester)

set(APP_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/test_glcm.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/test_ncc.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/test_tc.cc
)

add_executable(${PROJECT_NAME} ${APP_SOURCES})

include(GoogleTest)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/libs/common/include()
  ${CMAKE_SOURCE_DIR}/libs/glcm/include
  ${CMAKE_SOURCE_DIR}/libs/ncc/include
  ${CMAKE_SOURCE_DIR}/libs/tc-freq/include
  ${CMAKE_SOURCE_DIR}/libs/libmath/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  common
  glcm
  ncc
  tc-freq
  libmath
  GTest::gtest_main
)

target_compile_options(${PROJECT_NAME} PRIVATE 
  -Wall 
  -Werror
)