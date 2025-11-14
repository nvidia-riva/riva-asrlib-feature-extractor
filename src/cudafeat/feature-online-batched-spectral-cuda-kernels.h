// SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FEATURE_ONLINE_BATCHED_SPECTRAL_CUDA_KERNELS_H_
#define RIVA_ASRLIB_FEAT_FEATURE_ONLINE_BATCHED_SPECTRAL_CUDA_KERNELS_H_

#include "cu-feat-extr/cudafeat/lane-desc.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

void cuda_power_spectrum(int32_t max_chunk_frames, int32_t num_lanes,
                         int row_length, const float *A_in, int32_t ldi,
                         float *A_out, int32_t ldo, bool use_power, bool librosa_compat);

void cuda_mel_banks_compute(const LaneDesc *lanes, int32_t n_lanes,
                            int32_t max_chunk_frames, int32_t num_bins,
                            float energy_floor, int32_t *offsets,
                            int32_t *sizes, float **vecs, const float *feats,
                            int32_t ldf, float *mels, int32_t ldm,
                            bool use_log, bool librosa_compat);

void cuda_apply_lifter_and_floor_energy(const LaneDesc *lanes,
                                        int32_t num_lanes,
                                        int32_t max_chunk_frames, int num_cols,
                                        float cepstral_lifter, bool use_energy,
                                        float energy_floor, bool compat, float *log_energy,
                                        int32_t ldl, float *lifter_coeffs,
                                        float *features, int32_t ldf);

void cuda_process_window(const LaneDesc *lanes, int32_t num_lanes,
                         int32_t max_chunk_frames, int frame_length,
                         int padded_frame_length, bool librosa_center,
                         float dither, float energy_floor,
                         bool remove_dc_offset, float preemph_coeff,
                         bool need_raw_log_energy, float *log_energy_pre_window,
                         int32_t lde, const float *windowing,
                         float *tmp_windows, int32_t ldt, float *windows,
                         int32_t ldw);

void cuda_extract_window(const LaneDesc *lanes, int32_t num_lanes,
                         int32_t max_chunk_frames, int32_t frame_shift,
                         int32_t frame_length, int32_t frame_length_padded,
                         bool librosa_center, bool reflect_over_element,
                         RivaASRLibFeat_FrameRoundingStrategy frame_rnd_strat,
                         const float *wave, int32_t ldw,
                         float *windows, int32_t window_size, int32_t wlda,
                         float *stash, int32_t ssize, int32_t lds);

void cuda_dot_log(int32_t max_chunk_frames, int32_t num_lanes,
                  int32_t frame_length, float *signal_frame, int32_t lds,
                  float *signal_log_energy, int32_t lde);

void cuda_update_stash(const LaneDesc *lanes, int32_t num_lanes,
                       const float *wave, int32_t ldw, float *stash,
                       int32_t num_stash, int32_t lds);

} // namespace feat
} // namespace asrlib
} // namespace riva
#endif
