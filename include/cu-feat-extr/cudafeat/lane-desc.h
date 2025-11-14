// SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_LANE_DESC_H_
#define RIVA_ASRLIB_FEAT_LANE_DESC_H_

#include <cstdint>
#include "cu-feat-extr/c-cpp-bridge/pipeline-helper-types.h" // RivaASRLibFeat_ChannelId_t

namespace riva {
namespace asrlib {
namespace feat {

// The description for a single channel.
// A vector of these will be passed into components to
// control which channels are executed.
struct LaneDesc {
  RivaASRLibFeat_ChannelId_t channel;

  // number of samples in this chunk
  int32_t num_chunk_samples;

  // current sample for this chunk
  int32_t current_sample;

  // number of frames in this chunk.
  int32_t num_chunk_frames;

  // current frame for this chunk
  int32_t current_frame;

  // is this the last chunk
  int32_t last;

  // is this the first chunk
  int32_t first;
};

} // namespace feat
} // namespace asrlib
} // namespace riva
#endif
