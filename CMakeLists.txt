# Chose cmake version of latest nvidia kaldi container as of 5/21/2021
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(cuda_feature_extractor LANGUAGES CXX CUDA)

# cxx std;
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# cuda std
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA settings.
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_ARCH "sm_60")

# CUDA Flags
set(NVCC_FLAGS "--default-stream=per-thread")
list(APPEND NVCC_FLAGS "-Xcompiler=-fPIC")
list(APPEND NVCC_FLAGS "-arch=${CUDA_ARCH}")
# list(APPEND NVCC_FLAGS "-g")
# list(APPEND NVCC_FLAGS "-G")

# CUDA env defines
add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1)

# general include directories
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/src")

include_directories("${CMAKE_SOURCE_DIR}/c-cpp-bridge/include")
include_directories("${CMAKE_SOURCE_DIR}/c-cpp-bridge/src") # not utilized currently. No .h's in here.

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src/cudafeat)
add_subdirectory(src/cudamatrix)
add_subdirectory(src/feat)

add_subdirectory(c-cpp-bridge/src/c-options)
add_subdirectory(c-cpp-bridge/src/c-cudamatrix)

# Combine all object files into one library to provide to users.
add_library(riva-asrlib-cuda-feature-extractor
    STATIC
    $<TARGET_OBJECTS:cfe-cudafeat>
    $<TARGET_OBJECTS:cfe-cudamatrix>
    $<TARGET_OBJECTS:cfe-feat>

    # bridge targets
    $<TARGET_OBJECTS:cfe-c-options>
    $<TARGET_OBJECTS:cfe-c-cudamatrix>
)



