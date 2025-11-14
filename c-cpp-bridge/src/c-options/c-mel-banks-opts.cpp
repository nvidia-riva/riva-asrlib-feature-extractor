// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Phonexia s.r.o.;  Microsoft Corporation
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mel-banks-opts.h"

#include <cstdio>

extern "C" {

// Provides default values
void RivaASRLibFeat_MelBanksOptions_Create(RivaASRLibFeat_MelBanksOptions *opts, int32_t num_bins_in){

  // defaults
  opts->num_bins = 25;
  opts->low_freq = 20;
  opts->high_freq = 0;
  opts->vtln_low = 100;
  opts->vtln_high = -500;
  opts->debug_mel = false;
  opts->htk_mode = false;

  // param
  opts->num_bins = num_bins_in;

}

void RivaASRLibFeat_MelBanksOptions_Destroy(RivaASRLibFeat_MelBanksOptions* opts){
  // nothing for now.
}

void RivaASRLibFeat_MelBanksOptions_Print(const RivaASRLibFeat_MelBanksOptions *opts){

  printf("Mel Banks Options:\n");
  printf("num_bins: %i\n", opts->num_bins);
  printf("low_freq: %f\n", opts->low_freq);
  printf("high_freq: %f\n", opts->high_freq);
  printf("vtln_low: %f\n", opts->vtln_low);
  printf("vtln_high: %f\n", opts->vtln_high);
  printf("debug_mel: %d\n", opts->debug_mel);
  printf("htk_mode: %d\n", opts->htk_mode);
  printf("\n");
}

} // extern C

