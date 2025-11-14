// SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FEATURE_WINDOW_CUDA_H_
#define RIVA_ASRLIB_FEAT_FEATURE_WINDOW_CUDA_H_

#include "cu-feat-extr/cudamatrix/cu-matrix.h"
#include "cu-feat-extr/cudamatrix/cu-vector.h"
#include "cu-feat-extr/feat/feature-window.h"

namespace riva {
namespace asrlib {
namespace feat {

// This struct stores a feature window on the device.
// Behind the scense it just computes a feature window on
// the host and then copies it into device memory.
struct CudaFeatureWindowFunction {
  CudaFeatureWindowFunction() {}
  explicit CudaFeatureWindowFunction(const RivaASRLibFeat_FrameExtractionOptions &opts);
  CuVector<float> cu_window;
};

} // namespace feat
} // namespace asrlib
} // namespace riva

#endif  // RIVA_ASRLIB_FEAT_FEATURE_WINDOW_CUDA_H_
