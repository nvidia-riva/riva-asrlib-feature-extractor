// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely
// SPDX-FileCopyrightText: Copyright (c) 2012-2015  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_CU_DEVICE_H_
#define RIVA_ASRLIB_FEAT_CU_DEVICE_H_

#include <cublas_v2.h>
#include <curand.h>
#include <random>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cu-feat-extr/cudamatrix/cu-common.h"

namespace riva {
namespace asrlib {
namespace feat {

class CuDevice {
 public:

  static CuDevice& GetInstanceForThread() {
    // scott myers singleton, but per thread.
    static thread_local CuDevice this_thread_device;
    return this_thread_device;
  }

  cublasHandle_t GetCublasHandle() { return cublas_handle_; }
  curandGenerator_t GetCurandHandle() { return curand_handle_; }


  void SeedGpu() {
      std::default_random_engine generator;
      std::uniform_int_distribution<int> distribution(128, RAND_MAX);

      CURAND_SAFE_CALL(curandSetPseudoRandomGeneratorSeed(
            curand_handle_, distribution(generator)));
      CURAND_SAFE_CALL(curandSetGeneratorOffset(curand_handle_, 0));
  }

  CuDevice(const CuDevice &other) = delete;
  CuDevice(CuDevice &&other) = delete;

  CuDevice& operator=(const CuDevice &other) = delete;
  CuDevice& operator=(CuDevice &&other) = delete;

 private:

  CuDevice();
  ~CuDevice();

  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_handle_;

}; // class CuDevice


inline cublasHandle_t GetCublasHandle() {
  return CuDevice::GetInstanceForThread().GetCublasHandle();
}

inline curandGenerator_t GetCurandHandle() {
  return CuDevice::GetInstanceForThread().GetCurandHandle();
}


} // namespace feat
} // namespace asrlib
} // namespace riva

#endif // RIVA_ASRLIB_FEAT_CU_DEVICE_H_
