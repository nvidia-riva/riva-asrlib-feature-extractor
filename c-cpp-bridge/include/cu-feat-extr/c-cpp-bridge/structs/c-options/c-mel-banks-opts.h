// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Phonexia s.r.o.;  Microsoft Corporation.
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey).
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_C_MEL_BANKS_OPTS_
#define RIVA_ASRLIB_FEAT_C_MEL_BANKS_OPTS_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {

  // e.g. 25; number of triangular bins
  int32_t num_bins;

  // e.g. 20; lower frequency cutoff
  float low_freq;

  // an upper frequency cutoff; 0 -> no cutoff, negative
  // ->added to the Nyquist frequency to get the cutoff.
  float high_freq;

  // vtln lower cutoff of warping function.
  float vtln_low;

  // vtln upper cutoff of warping function: if negative, added
  // to the Nyquist frequency to get the cutoff.
  float vtln_high;

  // htk_mode is a "hidden" config, it does not show up on command line.
  // Enables more exact compatibility with HTK, for testing purposes.  Affects
  // mel-energy flooring and reproduces a bug in HTK.
  bool debug_mel;

  bool htk_mode;

} RivaASRLibFeat_MelBanksOptions;

// Provides default values
void RivaASRLibFeat_MelBanksOptions_Create(RivaASRLibFeat_MelBanksOptions *opts, int32_t num_bins_in);
void RivaASRLibFeat_MelBanksOptions_Destroy(RivaASRLibFeat_MelBanksOptions *opts);
void RivaASRLibFeat_MelBanksOptions_Print(const RivaASRLibFeat_MelBanksOptions *opts);


#ifdef __cplusplus
}
#endif

#endif