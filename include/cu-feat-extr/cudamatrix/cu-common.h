// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely, Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_CU_COMMON_H_
#define RIVA_ASRLIB_FEAT_CU_COMMON_H_

#define CU1DBLOCK 256

#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>

#include <iostream>

#include <stdint.h>

namespace riva {
namespace asrlib {
namespace feat {

#define CU_SAFE_CALL(fun)                                                                          \
  {                                                                                                \
    int32_t ret;                                                                                     \
    if ((ret = (fun)) != 0) {                                                                      \
      std::cerr << "cudaError_t " << ret << " : \"" << cudaGetErrorString((cudaError_t)ret)        \
                << "\" returned from '" << #fun << "' in file" << __FILE__ << " line " << __LINE__ \
                << "'" << std::endl;                                                               \
      ;                                                                                            \
    }                                                                                              \
  }

#define CUFFT_SAFE_CALL(fun)                                                    \
  {                                                                             \
    int32_t ret;                                                                  \
    if ((ret = (fun)) != CUFFT_SUCCESS) {                                       \
      std::cerr << "cublasResult " << ret << " returned from '" << #fun << "'"; \
    }                                                                           \
  }

#define CUBLAS_SAFE_CALL(fun)                                                                 \
  {                                                                                           \
    int32_t ret;                                                                                \
    if ((ret = (fun)) != 0) {                                                                 \
      std::cerr << "cublasStatus_t " << ret << " : \""                                        \
                << ::riva::asrlib::feat::cublasGetStatusString((cublasStatus_t)ret) << "\" returned from '" << #fun \
                << "'";                                                                       \
    }                                                                                         \
  }

#define CURAND_SAFE_CALL(fun)                                                                 \
  {                                                                                           \
    int32_t ret;                                                                                \
    if ((ret = (fun)) != 0) {                                                                 \
      std::cerr << "curandStatus_t " << ret << " : \""                                        \
                << ::riva::asrlib::feat::curandGetStatusString((curandStatus_t)ret) << "\" returned from '" << #fun \
                << "'";                                                                       \
    }                                                                                         \
  }

const char* curandGetStatusString(curandStatus_t status);

const char* cublasGetStatusString(cublasStatus_t status);


} // namespace feat
} // namespace asrlib
} // namespace riva


#endif
