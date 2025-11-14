// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-fbank-opts.h"

#include <cstdio>

extern "C" {

void RivaASRLibFeat_FbankOptions_Create(RivaASRLibFeat_FbankOptions *opts){

  RivaASRLibFeat_FrameExtractionOptions_Create(&opts->frame_opts);
  RivaASRLibFeat_MelBanksOptions_Create(&opts->mel_opts, 23);
  opts->use_energy = false;
  opts->energy_floor = 0.0;
  opts->raw_energy = true;
  opts->htk_compat = false;
  opts->use_log_fbank = true;
  opts->use_power = true;
  opts->librosa_compat = false;

}

void RivaASRLibFeat_FbankOptions_Destroy(RivaASRLibFeat_FbankOptions* opts){
  RivaASRLibFeat_FrameExtractionOptions_Destroy(&opts->frame_opts);
  RivaASRLibFeat_MelBanksOptions_Destroy(&opts->mel_opts);
}

void RivaASRLibFeat_FbankOptions_Print(const RivaASRLibFeat_FbankOptions *opts){

  printf("Fbank Options:\n");
  RivaASRLibFeat_FrameExtractionOptions_Print(&opts->frame_opts);
  RivaASRLibFeat_MelBanksOptions_Print(&opts->mel_opts);
  printf("use_energy: %d\n", opts->use_energy);
  printf("energy_floor: %f\n", opts->energy_floor);
  printf("raw_energy: %d\n", opts->raw_energy);
  printf("htk_compat: %d\n", opts->htk_compat);
  printf("use_log_fbank: %d\n", opts->use_log_fbank);
  printf("use_power: %d\n", opts->use_power);
  printf("librosa_compat: %d\n", opts->librosa_compat);
  printf("\n");
}

} // extern "C"