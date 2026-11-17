cmake_minimum_required(VERSION 3.22)
project(synthetic_exocortex_bci_gateway CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(exocortex_cuda STATIC src/cuda_embedding_engine.cu)

add_executable(exocortex_gateway src/main.cpp)
target_link_libraries(exocortex_gateway PRIVATE exocortex_cuda)

add_executable(exocortex_test src/test_runner.cpp)
target_link_libraries(exocortex_test PRIVATE exocortex_cuda)
