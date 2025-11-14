// SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <nvToolsExt.h>
#include "cudafeat/feature-window-cuda.h"

namespace riva {
namespace asrlib {
namespace feat {

CudaFeatureWindowFunction::CudaFeatureWindowFunction(
    const RivaASRLibFeat_FrameExtractionOptions &opts) {
  int32_t frame_length = opts.window_size;

  // Create CPU feature window
  FeatureWindowFunction feature_window(opts);

  // Copy into GPU memory
  cu_window.Resize(frame_length, RivaASRLibFeat_kUndefined);
  cu_window.CopyFromVec(feature_window.window);
}
} // namespace feat
} // namespace asrlib
} // namespace riva
