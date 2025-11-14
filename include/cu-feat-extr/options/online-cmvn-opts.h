// SPDX-FileCopyrightText: Copyright (c) 2013   Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2014   Yanqing Sun, Junjie Wang, Daniel Povey, Korbinian Riedhammer
// SPDX-FileCopyrightText: Copyright (c) 2025   NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_ONLINE_CMVN_OPTS_H_
#define RIVA_ASRLIB_FEAT_ONLINE_CMVN_OPTS_H_

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-online-cmvn-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

struct OnlineCmvnOptions {

  RivaASRLibFeat_OnlineCmvnOptions c_opts;

  OnlineCmvnOptions(){
    RivaASRLibFeat_OnlineCmvnOptions_Create(&c_opts);
  }
  ~OnlineCmvnOptions(){
    RivaASRLibFeat_OnlineCmvnOptions_Destroy(&c_opts);
  }

  OnlineCmvnOptions(const OnlineCmvnOptions &other) = default;
  OnlineCmvnOptions(OnlineCmvnOptions &&other) = default;

  OnlineCmvnOptions& operator=(const OnlineCmvnOptions &other) = default;
  OnlineCmvnOptions& operator=(OnlineCmvnOptions &&other) = default;


};


} // namespace feat
} // namespace asrlib
} // namespace riva


#endif