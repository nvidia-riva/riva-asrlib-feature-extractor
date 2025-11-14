// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University.
// SPDX-FileCopyrightText: Copyright (c) 2014-2016  Johns Hopkins University (author: Daniel Povey).
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_C_MFCC_OPTS_H_
#define RIVA_ASRLIB_FEAT_C_MFCC_OPTS_H_

#include <stdint.h>
#include <stdbool.h>
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mel-banks-opts.h"

#ifdef __cplusplus
extern "C" {
#endif

/// MfccOptions contains basic options for computing MFCC features.
typedef struct {

  RivaASRLibFeat_FrameExtractionOptions frame_opts;

  // defaults the #mel-banks to 23 for the MFCC computations.
  // this seems to be common for 16khz-sampled data,
  // but for 8khz-sampled data, 15 may be better.
  RivaASRLibFeat_MelBanksOptions mel_opts;

  // e.g. 13: num cepstral coeffs, counting zero.
  int32_t num_ceps;

  // use energy; else C0
  bool use_energy;

  // 0 by default; set to a value like 1.0 or 0.1 if
  // you disable dithering.
  float energy_floor;

  // If true, compute energy before preemphasis and windowing
  bool raw_energy;

  // Scaling factor on cepstra for HTK compatibility.
  // if 0.0, no liftering is done.
  float cepstral_lifter;

  // if true, put energy/C0 last and introduce a factor of
  // sqrt(2) on C0 to be the same as HTK.
  bool htk_compat;

  // TODO: the effect of this setting makes us compatible with what?
  bool compat_feat;

} RivaASRLibFeat_MfccOptions;


// Provides default values
void RivaASRLibFeat_MfccOptions_Create(RivaASRLibFeat_MfccOptions *opts);
void RivaASRLibFeat_MfccOptions_Destroy(RivaASRLibFeat_MfccOptions *opts);
void RivaASRLibFeat_MfccOptions_Print(const RivaASRLibFeat_MfccOptions *opts);


#ifdef __cplusplus
}
#endif

#endif