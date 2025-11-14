// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
// SPDX-FileCopyrightText: Copyright (c) 2014-2016  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_MFCC_OPTS_H_
#define RIVA_ASRLIB_FEAT_MFCC_OPTS_H_

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mfcc-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

struct MfccOptions {

  RivaASRLibFeat_MfccOptions c_opts;

  MfccOptions(){
    RivaASRLibFeat_MfccOptions_Create(&c_opts);
  }
  ~MfccOptions(){
    RivaASRLibFeat_MfccOptions_Destroy(&c_opts);
  }

  MfccOptions(const MfccOptions &other) = default;
  MfccOptions(MfccOptions &&other) = default;

  MfccOptions& operator=(const MfccOptions &other) = default;
  MfccOptions& operator=(MfccOptions &&other) = default;


};


} // namespace feat
} // namespace asrlib
} // namespace riva

#endif