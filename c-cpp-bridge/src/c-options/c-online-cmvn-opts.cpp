// SPDX-FileCopyrightText: Copyright (c) 2013   Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2014   Yanqing Sun, Junjie Wang, Daniel Povey, Korbinian Riedhammer
// SPDX-FileCopyrightText: Copyright (c) 2025   NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-online-cmvn-opts.h"

extern "C" {

// Provides default values
void RivaASRLibFeat_OnlineCmvnOptions_Create(RivaASRLibFeat_OnlineCmvnOptions *opts){

  opts->cmn_window = 600;
  opts->speaker_frames = 600;
  opts->global_frames = 200;
  opts->normalize_mean = true;
  opts->normalize_variance = false;
}
void RivaASRLibFeat_OnlineCmvnOptions_Destroy(RivaASRLibFeat_OnlineCmvnOptions *opts){
  // nothing for now
}


} // extern C