// SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/cudafeat/online-batched-feature-pipeline-cuda.h"

#include "feat/feature-fbank.h"
#include "feat/online-feature.h"

#include "cudafeat/feature-online-batched-cmvn-cuda.h"
#include "cu-feat-extr/cudafeat/lane-desc.h"
#include "cu-feat-extr/c-cpp-bridge/pipeline-helper-types.h" // RivaASRLibFeat_ChannelId_t
#include "feat/feature-mfcc.h"
#include "cudamatrix/cu-device.h"

#include "cudafeat/feature-online-batched-spectral-cuda.h"
#include <cassert>

#include <stdexcept>

#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime_api.h>

namespace riva {
namespace asrlib {
namespace feat {

OnlineBatchedFeaturePipelineCuda::OnlineBatchedFeaturePipelineCuda(
    const OnlineBatchedFeatureInfo &info_in,
    int32_t max_chunk_size_samples, int32_t max_lanes, int32_t num_channels)
    : info_(info_in),
      cmvn_(NULL),
      max_chunk_size_samples_(max_chunk_size_samples),
      max_lanes_(max_lanes),
      num_channels_(num_channels) {
  spectral_feat_ = NULL;
  cmvn_ = NULL;

  CuDevice::GetInstanceForThread();

  switch(info_.c_opts.feature_type){
    case RivaASRLibFeat_MFCC:
      frame_opts_ = info_.c_opts.mfcc_opts.frame_opts;
      break;
    case RivaASRLibFeat_FBANK:
      frame_opts_ = info_.c_opts.fbank_opts.frame_opts;
      break;
    default:
      // Which ever base feature was requested is not currently supported
      assert(false);
  }

  if(frame_opts_.librosa_center){
      max_chunk_size_samples_ += frame_opts_.padded_window_size; // aka fft_length
  }

  // compute maximum chunk size for a given number of samples
  // round up because there may be additional context provided
  int32_t shift = frame_opts_.window_shift;
  max_chunk_size_frames_ = (max_chunk_size_samples_ + shift - 1) / shift;

  switch(info_.c_opts.feature_type){
    case RivaASRLibFeat_MFCC:
      spectral_feat_ = new CudaOnlineBatchedSpectralFeatures(
          CudaSpectralFeatureOptions(info_.c_opts.mfcc_opts),
          max_chunk_size_frames_, num_channels_, max_lanes_);
      break;
    case RivaASRLibFeat_FBANK:
      spectral_feat_ = new CudaOnlineBatchedSpectralFeatures(
          CudaSpectralFeatureOptions(info_.c_opts.fbank_opts),
          max_chunk_size_frames_, num_channels_, max_lanes_);
      break;
    default:
    // Which ever base feature was requested is not currently supported
      assert(false);
  }

  if (info_.c_opts.use_cmvn) {
    if (info_.GetGlobalCmvnStats().NumCols() == 0) {
      throw std::runtime_error("global_cmvn_stats for OnlineCmvn must be non-empty.\n");
    }
    OnlineCmvnState cmvn_state(info_.GetGlobalCmvnStats());
    CudaOnlineCmvnState cu_cmvn_state(cmvn_state);

    // TODO do we want to parameterize stats coarsening factor?
    // Setting this likely won't impact performance or accuracy
    // but will improve memory usage.  It's unclear where we
    // would want to register this parameter though.
    cmvn_ =
        new CudaOnlineBatchedCmvn(info_.c_opts.cmvn_opts, cu_cmvn_state, FeatureDim(),
                                  max_chunk_size_frames_, num_channels_, 1);
  }

  current_samples_stash_ = new int32_t[num_channels_];

  // allocated pinned memory for storing channel desc
  // NOTE: Will block, but assuming this is fine since we're in the constructor.
  cudaMallocHost(&h_lanes_, sizeof(LaneDesc) * max_lanes_);

  // allocate device memory
  // TODO: now there is no locking version of this. Does this matter? See free() location as well
  // TODO: change to async after l4t gets cuda11 support.
  CU_SAFE_CALL(cudaMalloc(&lanes_, sizeof(LaneDesc) * max_lanes_));

  cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
}

OnlineBatchedFeaturePipelineCuda::~OnlineBatchedFeaturePipelineCuda() {
  if (spectral_feat_ != NULL) delete spectral_feat_;
  if (cmvn_ != NULL) delete cmvn_;

  cudaFreeHost(h_lanes_);

  delete[] current_samples_stash_;

  // TODO: now there is no locking version of this.
  // TODO: change to async after l4t gets cuda11 support.
  CU_SAFE_CALL(cudaFree(lanes_));

  cudaEventDestroy(event_);
}


int32_t OnlineBatchedFeaturePipelineCuda::FeatureDim() {
  return spectral_feat_->Dim();
}

void OnlineBatchedFeaturePipelineCuda::PrintOptions() const {

  assert(spectral_feat_);
  spectral_feat_->cumfcc_opts_.PrintOptions();
}


void OnlineBatchedFeaturePipelineCuda::ComputeFeaturesBatched(
    int32_t num_lanes, const std::vector<RivaASRLibFeat_ChannelId_t> &channels,
    const std::vector<int32_t> &num_chunk_samples,
    const std::vector<bool> &first, const std::vector<bool> &last,
    float sample_freq, const CuMatrix<float> &cu_waves,
    CuMatrix<float> *input_features,
    std::vector<int32_t> *num_frames_computed) {
  assert(num_lanes <= max_lanes_);
  assert(num_lanes <= num_frames_computed->size());

  // Ensure that h_lanes_ is consumed before overwriting.
  cudaEventSynchronize(event_);

  /// for each lane copy input into pinned memory
  for (int32_t lane = 0; lane < num_lanes; lane++) {
    RivaASRLibFeat_ChannelId_t channel = channels[lane];
    assert(channel < num_channels_);
    assert(num_chunk_samples[lane] <= max_chunk_size_samples_);

    LaneDesc desc;
    desc.channel = channel;
    desc.last = last[lane];
    desc.first = first[lane];

    desc.current_sample = desc.first ? 0 : current_samples_stash_[channel];

    desc.num_chunk_samples = num_chunk_samples[lane];
    desc.current_frame = NumFrames(desc.current_sample, frame_opts_, false);

    // Compute total number of samples and frames
    int32_t num_samples = desc.current_sample + desc.num_chunk_samples;
    int32_t num_frames = NumFrames(num_samples, frame_opts_, desc.last);

    desc.num_chunk_frames = num_frames - desc.current_frame;

    // store desc in lane array
    h_lanes_[lane] = desc;

    // update current_sames stash
    current_samples_stash_[channel] =
        desc.current_sample + desc.num_chunk_samples;

    // write how many frames will be computed to output array
    (*num_frames_computed)[lane] = desc.num_chunk_frames;
  }

  cudaMemcpyAsync(lanes_, h_lanes_, sizeof(LaneDesc) * num_lanes,
                  cudaMemcpyHostToDevice, cudaStreamPerThread);

  // record event to know when copy is finished so that we don't overwrite
  // pinned array
  cudaEventRecord(event_, cudaStreamPerThread);

  assert(info_.c_opts.feature_type == RivaASRLibFeat_MFCC || info_.c_opts.feature_type == RivaASRLibFeat_FBANK);
  // Fbank called via the MFCC codepath
  // MFCC
  float vtln_warp = 1.0;
  spectral_feat_->ComputeFeaturesBatched(
      lanes_, num_lanes, cu_waves, sample_freq, vtln_warp, input_features);

  if (info_.c_opts.use_cmvn) {
    cmvn_->ComputeFeaturesBatched(num_lanes, lanes_, *input_features,
                                  input_features);
  }

}

} // namespace feat
} // namespace asrlib
} // namespace riva
