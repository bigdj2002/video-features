project(common)

set(APP_SOURCES 
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/program_options_lite.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/common_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sources/YuvReader.cpp
)

add_library(${PROJECT_NAME} ${APP_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall
  -Werror
)