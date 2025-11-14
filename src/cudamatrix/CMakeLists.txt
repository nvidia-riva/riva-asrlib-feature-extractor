add_library(cfe-cudamatrix
    OBJECT
    cu-common.cc
    cu-device.cc
    cu-matrix.cu
    cu-vector.cc
)

target_compile_options(cfe-cudamatrix PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)