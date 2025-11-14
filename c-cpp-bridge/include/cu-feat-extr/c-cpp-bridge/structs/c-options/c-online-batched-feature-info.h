// SPDX-FileCopyrightText: Copyright (c) 2013-2014   Johns Hopkins University (author: Daniel Povey).
// SPDX-FileCopyrightText: Copyright (c) 2025        NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_C_ONLINE_BATCHED_FEATURE_INFO_H_
#define RIVA_ASRLIB_FEAT_C_ONLINE_BATCHED_FEATURE_INFO_H_

#include <stdbool.h>

#include "cu-feat-extr/c-cpp-bridge/pipeline-helper-types.h" // RivaASRLibFeat_SpectralFeatureType
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-fbank-opts.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mfcc-opts.h"

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-online-cmvn-opts.h"
#include "cu-feat-extr/c-cpp-bridge/opaque-types.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {

  RivaASRLibFeat_SpectralFeatureType feature_type;
  RivaASRLibFeat_MfccOptions mfcc_opts;
  RivaASRLibFeat_FbankOptions fbank_opts;

  bool use_cmvn;
  RivaASRLibFeat_OnlineCmvnOptions cmvn_opts;
  RivaASRLibFeat_CuMatrixD_t global_cmvn_stats;

} RivaASRLibFeat_OnlineBatchedFeatureInfo;

void RivaASRLibFeat_OnlineBatchedFeatureInfo_Create(
  RivaASRLibFeat_OnlineBatchedFeatureInfo *info,
  RivaASRLibFeat_SpectralFeatureType feature_type_in,
  bool use_cmvn_in
);

void RivaASRLibFeat_OnlineBatchedFeatureInfo_CreateFromOther(
  RivaASRLibFeat_OnlineBatchedFeatureInfo *info,
  const RivaASRLibFeat_OnlineBatchedFeatureInfo *other
);

void RivaASRLibFeat_OnlineBatchedFeatureInfo_CopyFromOther(
  RivaASRLibFeat_OnlineBatchedFeatureInfo *info,
  const RivaASRLibFeat_OnlineBatchedFeatureInfo *other
);

void RivaASRLibFeat_OnlineBatchedFeatureInfo_Destroy(
  RivaASRLibFeat_OnlineBatchedFeatureInfo *info
);


#ifdef __cplusplus
}
#endif

#endif // RIVA_ASRLIB_FEAT_C_ONLINE_BATCHED_FEATURE_INFO_H_
