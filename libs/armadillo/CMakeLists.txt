project(armadillo)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(/usr/lib/x86_64-linux-gnu)

add_library(${PROJECT_NAME} SHARED IMPORTED)
set_target_properties(${PROJECT_NAME} PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/libarmadillo.so.12.8.2"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

if(LAPACK_FOUND AND BLAS_FOUND)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()