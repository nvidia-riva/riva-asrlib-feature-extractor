add_library(cfe-cudafeat
    OBJECT
    feature-online-batched-cmvn-cuda-kernels.cu
    feature-online-batched-cmvn-cuda.cc
    feature-online-batched-spectral-cuda-kernels.cu
    feature-online-batched-spectral-cuda.cc
    feature-window-cuda.cu
    online-batched-feature-pipeline-cuda.cc
)

target_compile_options(cfe-cudafeat PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)