// SPDX-FileCopyrightText: Copyright (c) 2013-2014   Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025        NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-online-batched-feature-info.h"
#include "cu-feat-extr/c-cpp-bridge/c-cudamatrix/c-cu-matrix.h"

extern "C" {


void RivaASRLibFeat_OnlineBatchedFeatureInfo_Create(
        RivaASRLibFeat_OnlineBatchedFeatureInfo *info,
        RivaASRLibFeat_SpectralFeatureType feature_type_in,
        bool use_cmvn_in){

    info->feature_type = feature_type_in;
    RivaASRLibFeat_MfccOptions_Create(&info->mfcc_opts);
    RivaASRLibFeat_FbankOptions_Create(&info->fbank_opts);

    info->use_cmvn = use_cmvn_in;
    RivaASRLibFeat_OnlineCmvnOptions_Create(&info->cmvn_opts);
    RivaASRLibFeat_CuMatrixD_t_Create(&info->global_cmvn_stats);
}

void RivaASRLibFeat_OnlineBatchedFeatureInfo_CreateFromOther(
        RivaASRLibFeat_OnlineBatchedFeatureInfo *info,
        const RivaASRLibFeat_OnlineBatchedFeatureInfo *other){

    info->feature_type = other->feature_type;
    info->mfcc_opts = other->mfcc_opts;
    info->fbank_opts = other->fbank_opts;
    info->use_cmvn = other->use_cmvn;
    info->cmvn_opts = other->cmvn_opts;
    RivaASRLibFeat_CuMatrixD_t_CreateFromOther(
        &info->global_cmvn_stats, &other->global_cmvn_stats);
}

void RivaASRLibFeat_OnlineBatchedFeatureInfo_CopyFromOther(
        RivaASRLibFeat_OnlineBatchedFeatureInfo *info,
        const RivaASRLibFeat_OnlineBatchedFeatureInfo *other){

    if(info == other){
        return;
    }
    RivaASRLibFeat_OnlineBatchedFeatureInfo_Destroy(info);
    RivaASRLibFeat_OnlineBatchedFeatureInfo_CreateFromOther(info, other);
}

void RivaASRLibFeat_OnlineBatchedFeatureInfo_Destroy(
        RivaASRLibFeat_OnlineBatchedFeatureInfo *info) {

    RivaASRLibFeat_MfccOptions_Destroy(&info->mfcc_opts);
    RivaASRLibFeat_FbankOptions_Destroy(&info->fbank_opts);
    RivaASRLibFeat_OnlineCmvnOptions_Destroy(&info->cmvn_opts);
    RivaASRLibFeat_CuMatrixD_t_Destroy(&info->global_cmvn_stats);
}


} // extern "C"