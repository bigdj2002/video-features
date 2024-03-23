project(feature_generator)

set(APP_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp  
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/app.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/gop_distributor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/yuv_reader.cpp
)

add_executable(${PROJECT_NAME} ${APP_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/libs/common/include()
  ${CMAKE_SOURCE_DIR}/libs/glcm/include
  ${CMAKE_SOURCE_DIR}/libs/ncc/include
  ${CMAKE_SOURCE_DIR}/libs/tc-freq/include
  ${CMAKE_SOURCE_DIR}/libs/libmath/include
  ${JSONCPP_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  common
  glcm
  ncc
  tc-freq
  libmath
  -pthread
  ${JSONCPP_LIBRARIES}
)

target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall
  -Werror
)