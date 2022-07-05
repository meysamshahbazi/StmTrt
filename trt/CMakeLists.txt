cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(TensorRTSample)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

# TARGETS --------------------------------------------------------------------------------------------------------------
add_executable(head head.cpp)
target_include_directories(head PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(head PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})


# cmake_minimum_required(VERSION 2.6)

# project(inception)

# add_definitions(-std=c++14)

# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Release)

# include_directories(${PROJECT_SOURCE_DIR}/include)
# # include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# # cuda
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
# # tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)

# add_executable(inception ${PROJECT_SOURCE_DIR}/backbone_q.cpp)
# target_link_libraries(inception nvinfer)
# target_link_libraries(inception cudart)

# add_definitions(-O2 -pthread)


