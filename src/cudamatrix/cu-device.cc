// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely
// SPDX-FileCopyrightText: Copyright (c)      2013  Lucas Ondel
// SPDX-FileCopyrightText: Copyright (c) 2013-2015  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c)      2015  Guoguo Chen
// SPDX-FileCopyrightText: Copyright (c)      2025  NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>

#include "cudamatrix/cu-device.h"

namespace riva {
namespace asrlib {
namespace feat {

CuDevice::CuDevice() {

    CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_));
    CUBLAS_SAFE_CALL(cublasSetStream(cublas_handle_, cudaStreamPerThread));

    CURAND_SAFE_CALL(curandCreateGenerator(&curand_handle_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_SAFE_CALL(curandSetGeneratorOrdering(curand_handle_, CURAND_ORDERING_PSEUDO_DEFAULT));
    CURAND_SAFE_CALL(curandSetStream(curand_handle_, cudaStreamPerThread));
    SeedGpu();
}

CuDevice::~CuDevice() {

  if (cublas_handle_){
    CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_));
  }

  if (curand_handle_) {
    CURAND_SAFE_CALL(curandDestroyGenerator(curand_handle_));
  }
}


} // namespace feat
} // namespace asrlib
} // namespace riva
