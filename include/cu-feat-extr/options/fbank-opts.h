// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FBANK_OPTS_H_
#define RIVA_ASRLIB_FEAT_FBANK_OPTS_H_

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-fbank-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

struct FbankOptions {

  RivaASRLibFeat_FbankOptions c_opts;

  FbankOptions(){
    RivaASRLibFeat_FbankOptions_Create(&c_opts);
  }
  ~FbankOptions(){
    RivaASRLibFeat_FbankOptions_Destroy(&c_opts);
  }

  FbankOptions(const FbankOptions &other) = default;
  FbankOptions(FbankOptions &&other) = default;

  FbankOptions& operator=(const FbankOptions &other) = default;
  FbankOptions& operator=(FbankOptions &&other) = default;

};

} // namespace feat
} // namespace asrlib
} // namespace riva


#endif