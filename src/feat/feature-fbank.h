// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FEATURE_FBANK_H_
#define RIVA_ASRLIB_FEAT_FEATURE_FBANK_H_

#include <map>
#include <string>

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-fbank-opts.h"
#include "cu-feat-extr/feat/feature-window.h"
#include "feat/mel-computations.h"

namespace riva {
namespace asrlib {
namespace feat {

// Class for computing mel-filterbank features; see \ref feat_mfcc for more
// information.
class FbankComputer {
 public:

  explicit FbankComputer(const RivaASRLibFeat_FbankOptions &opts){
    RivaASRLibFeat_FbankOptions_Create(&opts_);
    opts_ = opts;
  }

  ~FbankComputer(){
    RivaASRLibFeat_FbankOptions_Destroy(&opts_);
  }

  const RivaASRLibFeat_FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  // Disallow copy/move
  FbankComputer& operator=(const FbankComputer &other) = delete;
  FbankComputer& operator=(FbankComputer &&other) = delete;
  FbankComputer(const FbankComputer &other) = delete;
  FbankComputer(FbankComputer &&other) = delete;

 private:

  RivaASRLibFeat_FbankOptions opts_;
};

} // namespace feat
} // namespace asrlib
} // namespace riva


#endif  // RIVA_ASRLIB_FEAT_FEATURE_FBANK_H_
