// SPDX-FileCopyrightText: Copyright (c) 2013   Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2014   Yanqing Sun, Junjie Wang, Daniel Povey, Korbinian Riedhammer
// SPDX-FileCopyrightText: Copyright (c) 2025   NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_C_ONLINE_CMVN_OPTS_H_
#define RIVA_ASRLIB_FEAT_C_ONLINE_CMVN_OPTS_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Note the similarity with SlidingWindowCmnOptions, but there
// are also differences.  One which doesn't appear in the config
// itself, because it's a difference between the setups, is that
// in OnlineCmn, we carry over data from the previous utterance,
// or, if no previous utterance is available, from global stats,
// or, if previous utterances are available but the total amount
// of data is less than prev_frames, we pad with up to "global_frames"
// frames from the global stats.
typedef struct {

  int32_t cmn_window;

  // must be <= cmn_window
  int32_t speaker_frames;

  // must be <= speaker_frames.
  int32_t global_frames;

  // Must be true if normalize_variance==true.
  bool normalize_mean;

  bool normalize_variance;

} RivaASRLibFeat_OnlineCmvnOptions;


// Provides default values
void RivaASRLibFeat_OnlineCmvnOptions_Create(RivaASRLibFeat_OnlineCmvnOptions *opts);
void RivaASRLibFeat_OnlineCmvnOptions_Destroy(RivaASRLibFeat_OnlineCmvnOptions *opts);


#ifdef __cplusplus
}
#endif

#endif