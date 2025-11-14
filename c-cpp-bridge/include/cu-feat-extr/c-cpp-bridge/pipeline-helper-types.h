// SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_PIPELINE_HELPER_TYPES_H_
#define RIVA_ASRLIB_FEAT_PIPELINE_HELPER_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef int32_t RivaASRLibFeat_ChannelId_t;

// NOTE: we do not support PLP opts
typedef enum {
    RivaASRLibFeat_MFCC,
    RivaASRLibFeat_FBANK
} RivaASRLibFeat_SpectralFeatureType;


#ifdef __cplusplus
}
#endif

#endif