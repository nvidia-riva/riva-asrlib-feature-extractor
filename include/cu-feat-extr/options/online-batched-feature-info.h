// SPDX-FileCopyrightText: Copyright (c) 2013-2014   Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025        NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_ONLINE_BATCHED_FEATURE_INFO_H_
#define RIVA_ASRLIB_FEAT_ONLINE_BATCHED_FEATURE_INFO_H_

#include <string>
#include <vector>
#include <cassert>

#include <stdexcept>

#include "cu-feat-extr/cudamatrix/cu-matrix.h"

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mfcc-opts.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-fbank-opts.h"
#include "cu-feat-extr/options/online-cmvn-opts.h"

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-online-batched-feature-info.h"

namespace riva {
namespace asrlib {
namespace feat {


/*
Stores configuration variables, including CMVN stats.
NOTE There is NO SUPPORT for the following:
  pitch extraction
  ivector
  OnlineSilenceWeightingConfig
  GetSamplingFrequency
*/
struct OnlineBatchedFeatureInfo {


  RivaASRLibFeat_OnlineBatchedFeatureInfo c_opts;

  // This constructor will be deprecated in a future release. Use the next one.
  OnlineBatchedFeatureInfo(const RivaASRLibFeat_OnlineBatchedFeatureInfo &opts_in) {
    RivaASRLibFeat_OnlineBatchedFeatureInfo_CreateFromOther(&c_opts, &opts_in);
  }

  OnlineBatchedFeatureInfo(RivaASRLibFeat_SpectralFeatureType feature_type_in, bool use_cmvn_in) {
    RivaASRLibFeat_OnlineBatchedFeatureInfo_Create(&c_opts, feature_type_in, use_cmvn_in);
  }

  ~OnlineBatchedFeatureInfo() {
    RivaASRLibFeat_OnlineBatchedFeatureInfo_Destroy(&c_opts);
  }

  OnlineBatchedFeatureInfo(const OnlineBatchedFeatureInfo &other){
    RivaASRLibFeat_OnlineBatchedFeatureInfo_CreateFromOther(&c_opts, &other.c_opts);
  }

  OnlineBatchedFeatureInfo(OnlineBatchedFeatureInfo &&other) = delete;

  OnlineBatchedFeatureInfo& operator=(const OnlineBatchedFeatureInfo &other){
    RivaASRLibFeat_OnlineBatchedFeatureInfo_CopyFromOther(&c_opts, &other.c_opts);
    return *this;
  }

  OnlineBatchedFeatureInfo& operator=(OnlineBatchedFeatureInfo &&other) = delete;

  // Performs reinterpret cast on opaque type automatically.
  CuMatrix<double>& GetGlobalCmvnStats(){
    return *reinterpret_cast< CuMatrix<double>* > (c_opts.global_cmvn_stats);
  }

  // Performs reinterpret cast on opaque type automatically.
  static CuMatrix<double>& GetGlobalCmvnStats(
      const RivaASRLibFeat_OnlineBatchedFeatureInfo &c_opts){
    return *reinterpret_cast< CuMatrix<double>* > (c_opts.global_cmvn_stats);
  }

};

} // namespace feat
} // namespace asrlib
} // namespace riva



#endif  // RIVA_ASRLIB_FEAT_ONLINE_BATCHED_FEATURE_INFO_H_
