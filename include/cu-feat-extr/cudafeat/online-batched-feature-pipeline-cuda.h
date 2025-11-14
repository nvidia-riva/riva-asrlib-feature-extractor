// SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_ONLINE_BATCHED_FEATURE_PIPELINE_CUDA_H_
#define RIVA_ASRLIB_FEAT_ONLINE_BATCHED_FEATURE_PIPELINE_CUDA_H_

#include <vector>


#include "cu-feat-extr/options/online-batched-feature-info.h"

#include "cu-feat-extr/cudafeat/lane-desc.h"
#include "cu-feat-extr/c-cpp-bridge/pipeline-helper-types.h" // RivaASRLibFeat_ChannelId_t

#include "cu-feat-extr/cudamatrix/cu-matrix.h"
#include "cu-feat-extr/cudamatrix/cu-vector.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

class CudaOnlineBatchedCmvn;
class CudaOnlineBatchedSpectralFeatures;

class OnlineBatchedFeaturePipelineCuda {
 public:
  explicit OnlineBatchedFeaturePipelineCuda(
      const OnlineBatchedFeatureInfo &info_in, int32_t max_chunk_size,
      int32_t max_lanes, int32_t num_channels);

  // Computes features for a batched chunk of audio data.
  // Upon exit this call guarentees that all input data is read
  // allowing the input arrays to be overwritten.
  //
  // All work is submitted to the per-thread-default stream and no
  // manual synchronization occurs outside of the per-thread-default-stream.
  //
  // if (num_lanes < max_lanes) only the valid lanes will be read and only
  // the valid lanes will be output.  Data in other lanes is undefined.
  //
  // if (num_chunk_samples[i] < max_chunk_size) only the valid samples
  // will be read by that lane.
  //
  // inputs:
  //   num_lanes:  number of lanes to compute featurs for
  //   channels:  lane vector specifying the channel for each lane
  //   num_chunk_samples:  lane vector specifying number of samples in each lane
  //     note: this cannot exceed max_chunk_size_samples
  //   first:  lane vector specifying if this is the first chunk of data
  //   last:  lane vector specifying if this is the last chunk of data
  //   sample_freq:  model sample frequency
  //   cu_waves:  lane matrix of input wave data
  //     with num rows equal to max_lanes
  //     and num cols equal to max_chunk_size
  // outputs:
  //   input_features:  lane matrix of output of base features
  //     with num rows equal to max_lanes * GetMaxChunkFrames()
  //     and num_cols equal to feat_dim
  //   num_frames_computed:  output vector containing the number of
  //     frames computed for each lane.

  void ComputeFeaturesBatched(int32_t num_lanes,
                              const std::vector<RivaASRLibFeat_ChannelId_t> &channels,
                              const std::vector<int32_t> &num_chunk_samples,
                              const std::vector<bool> &first,
                              const std::vector<bool> &last,
                              float sample_freq,
                              const CuMatrix<float> &cu_waves,
                              CuMatrix<float> *input_features,
                              std::vector<int32_t> *num_frames_computed);

  ~OnlineBatchedFeaturePipelineCuda();

  // Returns the maximum number of frames in a single chunk.
  // This should be used to size the input_features array that is
  // passed into ComputeFeaturesBatched
  int32_t GetMaxChunkFrames() { return max_chunk_size_frames_; }

  int32_t FeatureDim();

  const RivaASRLibFeat_FrameExtractionOptions &GetFrameOptions() { return frame_opts_; }

  void PrintOptions() const;

 private:
  OnlineBatchedFeatureInfo info_;

  CudaOnlineBatchedSpectralFeatures *spectral_feat_;
  CudaOnlineBatchedCmvn *cmvn_;
  RivaASRLibFeat_FrameExtractionOptions frame_opts_;

  int32_t max_chunk_size_samples_;  // The maximum size of a chunk in samples
  int32_t max_chunk_size_frames_;   // The maximum size of a chunk in frames
  int32_t max_lanes_;               // The maximum number of lanes
  int32_t num_channels_;            // The maximum number of channels

  // channel array for stashing sample count
  int32_t *current_samples_stash_;
  RivaASRLibFeat_ChannelId_t *channels_;

  // Host and Device array of lane descriptions
  LaneDesc *h_lanes_, *lanes_;

  cudaEvent_t event_;
};
} // namespace feat
} // namespace asrlib
} // namespace riva

#endif  // RIVA_ASRLIB_FEAT_ONLINE_BATCHED_FEATURE_PIPELINE_CUDA_H_
