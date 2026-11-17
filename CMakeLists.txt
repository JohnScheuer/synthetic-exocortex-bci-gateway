cmake_minimum_required(VERSION 3.22)
project(synthetic_exocortex_bci_gateway LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -mfma -ffast-math -Wall -Wextra -pthread")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})

add_library(exocortex_cuda src/cuda_embedding_engine.cu)
set_target_properties(exocortex_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(exocortex_gateway src/main.cpp)
target_link_libraries(exocortex_gateway PRIVATE exocortex_cuda CUDA::cudart pthread)

add_executable(exocortex_test src/test_runner.cpp)
target_link_libraries(exocortex_test PRIVATE exocortex_cuda CUDA::cudart pthread)
