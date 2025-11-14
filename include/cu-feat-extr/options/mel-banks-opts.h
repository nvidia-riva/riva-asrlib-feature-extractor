// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Phonexia s.r.o.;  Microsoft Corporation
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_MEL_BANKS_OPTS_
#define RIVA_ASRLIB_FEAT_MEL_BANKS_OPTS_

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mel-banks-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

struct MelBanksOptions {

  RivaASRLibFeat_MelBanksOptions c_opts;

  explicit MelBanksOptions(int num_bins) {
    RivaASRLibFeat_MelBanksOptions_Create(&c_opts, num_bins);
  }
  ~MelBanksOptions(){
    RivaASRLibFeat_MelBanksOptions_Destroy(&c_opts);
  }

  MelBanksOptions(const MelBanksOptions &other) = default;
  MelBanksOptions(MelBanksOptions &&other) = default;

  MelBanksOptions& operator=(const MelBanksOptions &other) = default;
  MelBanksOptions& operator=(MelBanksOptions &&other) = default;


};

} // namespace feat
} // namespace asrlib
} // namespace riva


#endif