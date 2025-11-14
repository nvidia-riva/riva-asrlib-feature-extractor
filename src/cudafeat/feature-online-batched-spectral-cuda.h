// SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FEATURE_BATCHED_SPECTRAL_CUDA_H_
#define RIVA_ASRLIB_FEAT_FEATURE_BATCHED_SPECTRAL_CUDA_H_

#include <cufft.h>

#include "cudafeat/feature-window-cuda.h"
#include "cu-feat-extr/cudafeat/lane-desc.h"
#include "cu-feat-extr/cudamatrix/cu-matrix.h"
#include "cu-feat-extr/cudamatrix/cu-vector.h"
#include "feat/feature-mfcc.h"

#include "cu-feat-extr/c-cpp-bridge/pipeline-helper-types.h" // RivaASRLibFeat_SpectralFeatureType
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mfcc-opts.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-fbank-opts.h"
#include <stdint.h>
#include <cassert>

namespace riva {
namespace asrlib {
namespace feat {

struct CudaSpectralFeatureOptions {
  RivaASRLibFeat_MfccOptions mfcc_opts;
  bool use_log_fbank; // LDB: Adding these two to enable fbank and mfcc
  bool use_power;     //   to use the same code path for GPU (and CPU?)
  bool use_dct;       // LDB: Adding this so that fbank can run w/o applying dct
  bool librosa_compat; // see c-bank-opts.h for doc.

  RivaASRLibFeat_SpectralFeatureType feature_type;

  explicit CudaSpectralFeatureOptions(RivaASRLibFeat_MfccOptions opts_in){
    RivaASRLibFeat_MfccOptions_Create(&mfcc_opts);
    mfcc_opts = opts_in;
    use_log_fbank = true;
    use_power = true;
    use_dct = true;
    librosa_compat = false;
    feature_type = RivaASRLibFeat_MFCC;
  }

  explicit CudaSpectralFeatureOptions(RivaASRLibFeat_FbankOptions opts){
    RivaASRLibFeat_MfccOptions_Create(&mfcc_opts);
    mfcc_opts.frame_opts      = opts.frame_opts;
    mfcc_opts.mel_opts        = opts.mel_opts;
    mfcc_opts.use_energy      = opts.use_energy;
    mfcc_opts.energy_floor    = opts.energy_floor;
    mfcc_opts.raw_energy      = opts.raw_energy;
    mfcc_opts.htk_compat      = opts.htk_compat;
    mfcc_opts.cepstral_lifter = 0.0f;
    use_log_fbank = opts.use_log_fbank;
    use_power = opts.use_power;
    use_dct = false;
    librosa_compat = opts.librosa_compat;
    feature_type = RivaASRLibFeat_FBANK;

    if(librosa_compat){
      assert(use_log_fbank);
    }
  }
  // Default is MFCC
  CudaSpectralFeatureOptions(){
    RivaASRLibFeat_MfccOptions_Create(&mfcc_opts);
    use_log_fbank = true;
    use_power = true;
    use_dct = true;
    librosa_compat = false;
    feature_type = RivaASRLibFeat_MFCC;
  }

  ~CudaSpectralFeatureOptions(){
    RivaASRLibFeat_MfccOptions_Destroy(&mfcc_opts);
  }

  void PrintOptions() const {
    printf("Cuda Spectral Feature Options:\n");
    RivaASRLibFeat_MfccOptions_Print(&mfcc_opts);
    printf("use_log_fbank: %d\n", use_log_fbank);
    printf("use_power: %d\n", use_power);
    printf("use_dct: %d\n", use_dct);
    printf("feature type: %s", feature_type == RivaASRLibFeat_MFCC ? "mfcc" : "fbank");
    printf("\n");
  }

  // Will need specialized versions if RivaASRLibFeat_MfccOptions ever defines _CopyFromOther() or _CreateFromOther()
  CudaSpectralFeatureOptions(const CudaSpectralFeatureOptions &other) = default;
  CudaSpectralFeatureOptions(CudaSpectralFeatureOptions &&other) = default;
  CudaSpectralFeatureOptions& operator=(const CudaSpectralFeatureOptions &other) = default;
  CudaSpectralFeatureOptions& operator=(CudaSpectralFeatureOptions &&other) = default;

};

// This class implements MFCC and Fbank computation in CUDA.
// It handles batched input.
// It takes input from device memory and outputs to
// device memory.  It also does no synchronization.
class CudaOnlineBatchedSpectralFeatures : public MfccComputer {
 public:
  void ComputeFeatures(const CuVector<float> &cu_wave,
                       float sample_freq, float vtln_warp,
                       CuMatrix<float> *cu_features) {
     // Non-batched processing not allowed from
     //    CudaOnlineBatchedSpectralFeatures
     assert(false);
  }

  void ComputeFeaturesBatched(const LaneDesc *lanes, int32_t n_lanes,
                              const CuMatrix<float> &cu_wave_in,
                              float sample_freq, float vtln_warp,
                              CuMatrix<float> *cu_feats_out);

  CudaOnlineBatchedSpectralFeatures(const CudaSpectralFeatureOptions &opts,
                                    int32_t max_chunk_frames,
                                    int32_t num_channels, int32_t max_lanes);
  ~CudaOnlineBatchedSpectralFeatures();
  CudaSpectralFeatureOptions cumfcc_opts_;
  int32_t Dim()
  // The dimension of the output is different for MFCC and Fbank.
  // This returns the appropriate value depending on the feature
  // extraction algorithm
  {
    if (cumfcc_opts_.feature_type == RivaASRLibFeat_MFCC) return MfccComputer::Dim();
    // If we're running fbank, we need to set the dimension right
    else
      return cumfcc_opts_.mfcc_opts.mel_opts.num_bins +
             (cumfcc_opts_.mfcc_opts.use_energy ? 1 : 0);
  }

 private:

  void ExtractWindowsBatched(const LaneDesc *lanes, int32_t num_lanes,
                             const CuMatrix<float> &wave);

  void UpdateStashBatched(const LaneDesc *lanes, int32_t num_lanes,
                          const CuMatrix<float> &wave);

  void ProcessWindowsBatched(const LaneDesc *lanes, int32_t num_lanes,
                             const RivaASRLibFeat_FrameExtractionOptions &opts,
                             CuMatrix<float> *log_energy_pre_window);

  void ComputeFinalFeaturesBatched(const LaneDesc *lanes, int32_t num_lanes,
                                   float vtln_wrap,
                                   CuMatrix<float> *cu_signal_log_energy,
                                   CuMatrix<float> *cu_features);

  CuVector<float> cu_lifter_coeffs_;
  CuMatrix<float> cu_windows_;
  CuMatrix<float> tmp_window_, cu_mel_energies_;
  CuMatrix<float> cu_dct_matrix_;
  CuMatrix<float> stash_;
  CuMatrix<float> power_spectrum_;
  CuMatrix<float> raw_log_energies_;

  int frame_length_, padded_length_, fft_length_, fft_batch_size_;
  cufftHandle plan_;
  CudaFeatureWindowFunction window_function_;

  int bin_size_;
  int32_t *offsets_, *sizes_;
  CuVector<float> *cu_vecs_;
  float **vecs_;

  // for sanity checking cufft
  int32_t stride_, tmp_stride_;

  int32_t max_chunk_frames_;
  int32_t num_channels_;
  int32_t max_lanes_;
};
} // namespace feat
} // namespace asrlib
} // namespace riva

#endif
