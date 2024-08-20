cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(stmtracker VERSION 0.0.0 LANGUAGES C CXX)


# find_package(Threads)
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_86,code=sm_86)

SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -Wunused-function")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )


file(GLOB CLASS_SRC 
        "*.cpp"
        "*.cu"
    )
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

# TARGETS --------------------------------------------------------------------------------------------------------------
cuda_add_executable(${PROJECT_NAME}  ${CLASS_SRC})
target_include_directories(${PROJECT_NAME}  PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}   ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME}   ${CUDA_LIBRARIES} )

target_link_libraries(${PROJECT_NAME}   ${TensorRT_LIBRARIES})

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} )


