// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
// SPDX-FileCopyrightText: Copyright (c) 2014-2016  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mfcc-opts.h"

#include <cstdio>

extern "C" {

void RivaASRLibFeat_MfccOptions_Create(RivaASRLibFeat_MfccOptions *opts){

  RivaASRLibFeat_FrameExtractionOptions_Create(&opts->frame_opts);
  RivaASRLibFeat_MelBanksOptions_Create(&opts->mel_opts, 23);

  opts->num_ceps = 13;
  opts->use_energy = true;
  opts->energy_floor = 0.0;
  opts->raw_energy = true;
  opts->cepstral_lifter = 22.0;
  opts->htk_compat = false;
  opts->compat_feat = false;
}

void RivaASRLibFeat_MfccOptions_Destroy(RivaASRLibFeat_MfccOptions* opts){
  RivaASRLibFeat_FrameExtractionOptions_Destroy(&opts->frame_opts);
  RivaASRLibFeat_MelBanksOptions_Destroy(&opts->mel_opts);
}

void RivaASRLibFeat_MfccOptions_Print(const RivaASRLibFeat_MfccOptions *opts){
  printf("Mfcc Options: \n");

  RivaASRLibFeat_FrameExtractionOptions_Print(&opts->frame_opts);
  RivaASRLibFeat_MelBanksOptions_Print(&opts->mel_opts);
  printf("num_ceps: %i\n", opts->num_ceps);
  printf("use_energy: %d\n", opts->use_energy);
  printf("energy_floor: %f\n", opts->energy_floor);
  printf("raw_energy: %d\n", opts->raw_energy);
  printf("cepstral_lifter: %f\n", opts->cepstral_lifter);
  printf("htk_compat: %d\n", opts->htk_compat);
  printf("compat_feat: %d\n", opts->compat_feat);
  printf("\n");
}

} // extern "C"