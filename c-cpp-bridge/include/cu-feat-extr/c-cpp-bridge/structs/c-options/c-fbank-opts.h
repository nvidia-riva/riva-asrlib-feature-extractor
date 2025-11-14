// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely.
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey).
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_C_FBANK_OPTS_H_
#define RIVA_ASRLIB_FEAT_C_FBANK_OPTS_H_

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mel-banks-opts.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// FbankOptions contains basic options for computing filterbank features.
/// It only includes things that can be done in a "stateless" way, i.e.
/// it does not include energy max-normalization.
/// It does not include delta computation.
typedef struct {

  RivaASRLibFeat_FrameExtractionOptions frame_opts;

  // defaults the #mel-banks to 23 for the FBANK computations.
  // this seems to be common for 16khz-sampled data,
  // but for 8khz-sampled data, 15 may be better.
  RivaASRLibFeat_MelBanksOptions mel_opts;

  // append an extra dimension with energy to the filter banks
  bool use_energy;

  float energy_floor;

  // If true, compute energy before preemphasis and windowing
  bool raw_energy;

  // If true, put energy last (if using energy)
  bool htk_compat;

  // if true (default), produce log-filterbank, else linear
  bool use_log_fbank;

  // if true (default), use power in filterbank analysis, else magnitude.
  bool use_power;

  // iff enabled:
  // Will not use htk approx for mel<->hz conversion, will apply slaney normalization, and
  // will always apply floor.
  bool librosa_compat;

} RivaASRLibFeat_FbankOptions;

// Provides default values
void RivaASRLibFeat_FbankOptions_Create(RivaASRLibFeat_FbankOptions *opts);

void RivaASRLibFeat_FbankOptions_Destroy(RivaASRLibFeat_FbankOptions *opts);

void RivaASRLibFeat_FbankOptions_Print(const RivaASRLibFeat_FbankOptions *opts);


#ifdef __cplusplus
}
#endif


#endif