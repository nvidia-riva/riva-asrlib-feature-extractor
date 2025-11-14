// SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cudafeat/feature-online-batched-spectral-cuda.h"
#include "cudafeat/feature-online-batched-spectral-cuda-kernels.h"
#include "cu-feat-extr/cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace riva {
namespace asrlib {
namespace feat {

CudaOnlineBatchedSpectralFeatures::CudaOnlineBatchedSpectralFeatures(
    const CudaSpectralFeatureOptions &opts, int32_t max_chunk_frames,
    int32_t num_channels, int32_t max_lanes)
    : MfccComputer(opts),

      // NOTE: using CopyFromVec below instead.
      // cu_lifter_coeffs_(lifter_coeffs_),
      cu_dct_matrix_(dct_matrix_),
      window_function_(opts.mfcc_opts.frame_opts),
      max_chunk_frames_(max_chunk_frames),
      num_channels_(num_channels),
      max_lanes_(max_lanes) {

  // NOTE: doing this instead of ctor above
  cu_lifter_coeffs_.CopyFromVec(lifter_coeffs_);

  assert(max_chunk_frames > 0);
  const MelBanks *mel_banks = GetMelBanks(1.0, opts.librosa_compat);
  const std::vector<std::pair<int32_t, std::vector<float>>> &bins =
      mel_banks->GetBins();
  int size = bins.size();
  bin_size_ = size;
  std::vector<int32_t> offsets(size), sizes(size);
  std::vector<float *> vecs(size);
  cu_vecs_ = new CuVector<float>[size];
  for (int i = 0; i < bins.size(); i++) {
    cu_vecs_[i].Resize(bins[i].second.size(), RivaASRLibFeat_kUndefined);
    cu_vecs_[i].CopyFromVec(bins[i].second);
    vecs[i] = cu_vecs_[i].Data();
    sizes[i] = cu_vecs_[i].Dim();
    offsets[i] = bins[i].first;
  }

  // NOTE: no locking version now.
  // TODO: change to async after l4t gets cuda11 support.
  cudaMalloc(&offsets_, size * sizeof(int32_t));
  cudaMalloc(&sizes_, size * sizeof(int32_t));
  cudaMalloc(&vecs_, size * sizeof(float *));

  CU_SAFE_CALL(cudaMemcpyAsync(vecs_, &vecs[0], size * sizeof(float *),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CU_SAFE_CALL(cudaMemcpyAsync(offsets_, &offsets[0], size * sizeof(int32_t),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CU_SAFE_CALL(cudaMemcpyAsync(sizes_, &sizes[0], size * sizeof(int32_t),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CU_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));

  const RivaASRLibFeat_FrameExtractionOptions frame_opts = opts.mfcc_opts.frame_opts;
  frame_length_ = frame_opts.window_size;
  padded_length_ = frame_opts.padded_window_size;
  fft_length_ = padded_length_ / 2;  // + 1;
  fft_batch_size_ = 800;

  // place holders to get strides for cufft.  these will be resized correctly
  // later. The +2 for cufft/fftw requirements of an extra element at the end.
  // Turning off stride because cufft seems buggy with a stride
  int32_t fft_num_frames_all_chunks = max_chunk_frames * max_lanes_;

  // Increase frame total to next multiple of fft_batch_size_
  int32_t fft_num_frames_all_chunks_padded = fft_num_frames_all_chunks + (fft_batch_size_ - fft_num_frames_all_chunks % fft_batch_size_);
  cu_windows_.Resize(fft_num_frames_all_chunks_padded, padded_length_, RivaASRLibFeat_kUndefined,
                     RivaASRLibFeat_kStrideEqualNumCols);
  //+1 matches cufft/fftw requirements
  tmp_window_.Resize(fft_num_frames_all_chunks_padded, padded_length_ + 2,
                     RivaASRLibFeat_kUndefined, RivaASRLibFeat_kStrideEqualNumCols);

  // Pre-allocated memory for power spectra
  power_spectrum_.Resize(max_chunk_frames_ * max_lanes_, padded_length_ / 2 + 1,
                         RivaASRLibFeat_kUndefined);
  raw_log_energies_.Resize(max_lanes_, max_chunk_frames_, RivaASRLibFeat_kUndefined);
  cu_mel_energies_.Resize(max_chunk_frames_ * max_lanes_, bin_size_,
                          RivaASRLibFeat_kUndefined);
  int32_t max_stash_size =
      2 * (frame_opts.window_size / 2 + frame_opts.window_shift);
  stash_.Resize(num_channels_, max_stash_size);

  stride_ = cu_windows_.Stride();
  tmp_stride_ = tmp_window_.Stride();

  cufftPlanMany(&plan_, 1, &padded_length_, NULL, 1, stride_, NULL, 1,
                tmp_stride_ / 2, CUFFT_R2C, fft_batch_size_);
  cufftSetStream(plan_, cudaStreamPerThread);
  cumfcc_opts_ = opts;
}

// ExtractWindow extracts a windowed frame of waveform with a power-of-two,
// padded size.  It does mean subtraction, pre-emphasis and dithering as
// requested.
void CudaOnlineBatchedSpectralFeatures::ExtractWindowsBatched(
    const LaneDesc *lanes, int32_t num_lanes,
    const CuMatrix<float> &wave) {
  CU_SAFE_CALL(cudaGetLastError());
  const RivaASRLibFeat_FrameExtractionOptions &opts = GetFrameOptions();
  cuda_extract_window(
      lanes, num_lanes, max_chunk_frames_, opts.window_shift,
      opts.window_size, opts.padded_window_size, opts.librosa_center,
      opts.reflect_over_element,
      opts.frame_rnd_strat, wave.Data(),
      wave.Stride(), cu_windows_.Data(), opts.window_size,
      cu_windows_.Stride(), stash_.Data(), stash_.NumCols(), stash_.Stride());
}

void CudaOnlineBatchedSpectralFeatures::ProcessWindowsBatched(
    const LaneDesc *lanes, int32_t num_lanes,
    const RivaASRLibFeat_FrameExtractionOptions &opts,
    CuMatrix<float> *log_energy_pre_window) {
  int fft_num_frames = cu_windows_.NumRows();
  assert(fft_num_frames % fft_batch_size_ == 0);

  cuda_process_window(
      lanes, num_lanes, max_chunk_frames_, frame_length_,
      padded_length_, opts.librosa_center, opts.dither,
      std::numeric_limits<float>::epsilon(), opts.remove_dc_offset,
      opts.preemph_coeff, NeedRawLogEnergy(), log_energy_pre_window->Data(),
      log_energy_pre_window->Stride(), window_function_.cu_window.Data(),
      tmp_window_.Data(), tmp_window_.Stride(), cu_windows_.Data(),
      cu_windows_.Stride());

  CU_SAFE_CALL(cudaGetLastError());
}

void CudaOnlineBatchedSpectralFeatures::UpdateStashBatched(
    const LaneDesc *lanes, int32_t num_lanes,
    const CuMatrix<float> &wave) {
  assert(stash_.NumCols() < 1024);

  cuda_update_stash(lanes, num_lanes, wave.Data(), wave.Stride(), stash_.Data(),
                    stash_.NumCols(), stash_.Stride());
}

void CudaOnlineBatchedSpectralFeatures::ComputeFinalFeaturesBatched(
    const LaneDesc *lanes, int32_t num_lanes, float vtln_wrap,
    CuMatrix<float> *cu_signal_log_energy,
    CuMatrix<float> *cu_features) {

  RivaASRLibFeat_MfccOptions mfcc_opts = cumfcc_opts_.mfcc_opts;
  CuVector<float> tmp;
  assert(mfcc_opts.htk_compat == false);

  if (num_lanes == 0) return;

  if (mfcc_opts.use_energy && !mfcc_opts.raw_energy) {
    cuda_dot_log(max_chunk_frames_, num_lanes, cu_windows_.NumCols(),
                 cu_windows_.Data(), cu_windows_.Stride(),
                 cu_signal_log_energy->Data(), cu_signal_log_energy->Stride());
    CU_SAFE_CALL(cudaGetLastError());
  }

  // make sure a reallocation hasn't changed these
  assert(cu_windows_.Stride() == stride_);
  assert(tmp_window_.Stride() == tmp_stride_);

  // Perform FFTs in batches of fft_size.  This reduces memory requirements
  for (int idx = 0; idx < max_chunk_frames_ * num_lanes;
       idx += fft_batch_size_) {
    CUFFT_SAFE_CALL(cufftExecR2C(
        plan_, cu_windows_.Data() + cu_windows_.Stride() * idx,
        (cufftComplex *)(tmp_window_.Data() + tmp_window_.Stride() * idx)));
  }

  // Compute Power spectrum
  const bool use_power = cumfcc_opts_.mfcc_opts.compat_feat ?
    false : cumfcc_opts_.use_power;

  cuda_power_spectrum(max_chunk_frames_, num_lanes, padded_length_,
                      tmp_window_.Data(), tmp_window_.Stride(),
                      power_spectrum_.Data(), power_spectrum_.Stride(),
                      use_power, cumfcc_opts_.librosa_compat);

  CU_SAFE_CALL(cudaGetLastError());

  int num_bins = bin_size_;

  // Not to be confused with the energy floor provided in fbank/mfcc option structs.
  const float mel_banks_compute_energy_floor = cumfcc_opts_.librosa_compat ?
    5.96046448e-8 /* 2**-24 used in NeMo */
    : std::numeric_limits<float>::epsilon();

  // mel banks plus optional dct transform
  if (cumfcc_opts_.use_dct) {
    assert(!cumfcc_opts_.librosa_compat); // librosa should not be performing dct
    // MFCC uses dct
    cuda_mel_banks_compute(lanes, num_lanes, max_chunk_frames_, num_bins,
                         mel_banks_compute_energy_floor, offsets_,
                         sizes_, vecs_, power_spectrum_.Data(),
                         power_spectrum_.Stride(), cu_mel_energies_.Data(),
                         cu_mel_energies_.Stride(), cumfcc_opts_.use_log_fbank,
                         cumfcc_opts_.librosa_compat);
    CU_SAFE_CALL(cudaGetLastError());

    cublasHandle_t cublas_handle = GetCublasHandle();

    if (cu_features->NumRows() > cu_mel_energies_.NumRows()) {

      // CuSubMatrix<float> cu_feats_sub(*cu_features, 0,
      //                                     cu_mel_energies_.NumRows(), 0,
      //                                     cu_features->NumCols());
      // cu_feats_sub.AddMatMat(1.0, cu_mel_energies_, RivaASRLibFeat_kNoTrans, cu_dct_matrix_,
      //                        RivaASRLibFeat_kTrans, 0.0);

      // NOTE using one specialized function below instead of implementing our own matrix sub-view code.

      cu_features->AddMatMatIgnoreDestNTrailingRows(
        1.0, cu_mel_energies_, RivaASRLibFeat_kNoTrans, cu_dct_matrix_, RivaASRLibFeat_kTrans, 0.0,
        cublas_handle, cu_features->NumRows() - cu_mel_energies_.NumRows());

    } else {
      cu_features->AddMatMat(1.0, cu_mel_energies_, RivaASRLibFeat_kNoTrans, cu_dct_matrix_,
                             RivaASRLibFeat_kTrans, 0.0, cublas_handle);
    }

    cuda_apply_lifter_and_floor_energy(
        lanes, num_lanes, max_chunk_frames_, cu_features->NumCols(),
        mfcc_opts.cepstral_lifter, mfcc_opts.use_energy,
        mfcc_opts.energy_floor, mfcc_opts.compat_feat,
        cu_signal_log_energy->Data(), cu_signal_log_energy->Stride(),
        cu_lifter_coeffs_.Data(), cu_features->Data(), cu_features->Stride());
    CU_SAFE_CALL(cudaGetLastError());
  } else {

    // fbank puts the result of mel_banks_compute directly into cu_features
    cuda_mel_banks_compute(lanes, num_lanes, max_chunk_frames_, num_bins,
                         mel_banks_compute_energy_floor, offsets_,
                         sizes_, vecs_, power_spectrum_.Data(),
                         power_spectrum_.Stride(), cu_features->Data(),
                         cu_features->Stride(), cumfcc_opts_.use_log_fbank,
                         cumfcc_opts_.librosa_compat);
    CU_SAFE_CALL(cudaGetLastError());
  }
  CU_SAFE_CALL(cudaGetLastError());
}

void CudaOnlineBatchedSpectralFeatures::ComputeFeaturesBatched(
    const LaneDesc *lanes, int32_t n_lanes,
    const CuMatrix<float> &cu_wave_in, float sample_freq,
    float vtln_warp, CuMatrix<float> *cu_feats_out) {
  // Note: cu_features is actually a rank 3 tensor.
  //       channels x frames x features
  // it is currently represented as a matrix with n_channels*n_frames rows and
  //                                              n_features cols
  const RivaASRLibFeat_FrameExtractionOptions &frame_opts = GetFrameOptions();

  if (frame_opts.dither != 0.0f) {
    // Calling cu-rand directly
    // CuRand class works on CuMatrixBase which must
    // assume that the matrix is part of a larger matrix
    // Doing this directly avoids unecessary memory copies
    CURAND_SAFE_CALL(
        curandGenerateNormal(GetCurandHandle(), tmp_window_.Data(),
                             tmp_window_.NumRows() * tmp_window_.Stride(),
                             0.0 /*mean*/, 1.0 /*stddev*/));
  }

  // Extract Windows
  ExtractWindowsBatched(lanes, n_lanes, cu_wave_in);

  UpdateStashBatched(lanes, n_lanes, cu_wave_in);

  // Process Windows
  ProcessWindowsBatched(lanes, n_lanes, frame_opts, &raw_log_energies_);

  // Compute Features
  ComputeFinalFeaturesBatched(lanes, n_lanes, 1.0, &raw_log_energies_, cu_feats_out);
}

CudaOnlineBatchedSpectralFeatures::~CudaOnlineBatchedSpectralFeatures() {
  delete[] cu_vecs_;
  // NOTE: no locking version.
  // TODO: change to async after l4t gets cuda11 support.
  cudaFree(vecs_);
  cudaFree(offsets_);
  cudaFree(sizes_);
  cufftDestroy(plan_);
}
} // namespace feat
} // namespace asrlib
} // namespace riva
