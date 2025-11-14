// SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FEATURE_ONLINE_BATCHED_CMVN_CUDA_H_
#define RIVA_ASRLIB_FEAT_FEATURE_ONLINE_BATCHED_CMVN_CUDA_H_

#include "cu-feat-extr/cudafeat/lane-desc.h"
#include "cu-feat-extr/cudamatrix/cu-matrix.h"
#include "feat/online-feature.h"
#include <stdint.h>
#include "cu-feat-extr/cudamatrix/cu-matrix.h"

namespace riva {
namespace asrlib {
namespace feat {

struct CudaOnlineCmvnState {
  // The following is the global CMVN stats, in the usual
  // format, of dimension 2 x (dim+1), as [  sum-stats          count
  //                                       sum-sqared-stats   0    ]
  CuMatrix<float> global_cmvn_stats;
  CuMatrix<float> speaker_cmvn_stats;

  // CudaOnlineCmvnState(){};
  CudaOnlineCmvnState(const OnlineCmvnState &cmvn_state)
      : global_cmvn_stats(cmvn_state.global_cmvn_stats),
      speaker_cmvn_stats(cmvn_state.speaker_cmvn_stats) {}
};

class CudaOnlineBatchedCmvn {
 public:
  CudaOnlineBatchedCmvn(const RivaASRLibFeat_OnlineCmvnOptions &opts,
                        const CudaOnlineCmvnState &cmvn_state, int32_t feat_dim,
                        int32_t chunk_size, int32_t num_channels,
                        int32_t stats_coarsening_factor);

  ~CudaOnlineBatchedCmvn();

  // Computes a chunk of features for each channel included in lanes
  void ComputeFeaturesBatched(int32_t num_lanes, const LaneDesc *lanes,
                              const CuMatrix<float> &feats_in,
                              CuMatrix<float> *feats_out);

 private:
  const RivaASRLibFeat_OnlineCmvnOptions &opts_;
  const CudaOnlineCmvnState cmvn_state_;

  int32_t feat_dim_;
  int32_t chunk_size_; // max chunk size frames
  int32_t num_channels_;

  // The number of frames for each fragment of stats.
  // Larger = faster, less memory, but less accurate
  // Smaller = slower, more memory, but more accurate,
  // 1 is equivalent to the non-batched version
  int32_t stats_coarsening_factor_;
  int32_t num_fragments_;  // window_size / stats_coarsening_factor_

  // This matrix stores prefix sum audio statistics in a rolling
  // buffer.  The stats are coarsened by stats_coarsening_factor_.
  // Coarsening reduces memory usage at a potential cost in
  // accuracy.  Matrix stores both sum and sum2 as float2 but
  // the matrix type is float as CuMatrix does not support float2.
  // val.x = sum and val.y = sum^2
  // Rows = channels, Cols = feat_dim * chunk_size/coarsening factor * 2
  CuMatrix<float> stats_fragments_;
};

} // namespace feat
} // namespace asrlib
} // namespace riva

#endif
