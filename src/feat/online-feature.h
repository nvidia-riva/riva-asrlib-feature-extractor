// SPDX-FileCopyrightText: Copyright (c) 2013   Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2014   Yanqing Sun, Junjie Wang, Daniel Povey, Korbinian Riedhammer
// SPDX-FileCopyrightText: Copyright (c) 2025   NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_ONLINE_FEATURE_H_
#define RIVA_ASRLIB_FEAT_ONLINE_FEATURE_H_

#include <string>
#include <vector>
#include <deque>
#include <stdint.h>

#include "cu-feat-extr/cudamatrix/cu-matrix.h"
#include "cu-feat-extr/options/online-cmvn-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

// /** Struct OnlineCmvnState stores the state of CMVN adaptation between
//     utterances (but not the state of the computation within an utterance).  It
//     stores the global CMVN stats and the stats of the current speaker (if we
//     have seen previous utterances for this speaker), and possibly will have a
//     member "frozen_state": if the user has called the function Freeze() of class
//     OnlineCmvn, to fix the CMVN so we can estimate fMLLR on top of the fixed
//     value of cmvn.  If nonempty, "frozen_state" will reflect how we were
//     normalizing the mean and (if applicable) variance at the time when that
//     function was called.
// */
struct OnlineCmvnState {
  // The following is the total CMVN stats for this speaker (up till now), in
  // the same format.
  CuMatrix<double> speaker_cmvn_stats;

  // The following is the global CMVN stats, in the usual
  // format, of dimension 2 x (dim+1), as [  sum-stats          count
  //                                       sum-squared-stats   0    ]
  CuMatrix<double> global_cmvn_stats;

  // If nonempty, contains CMVN stats representing the "frozen" state
  // of CMVN that reflects how we were normalizing the data when the
  // user called the Freeze() function in class OnlineCmvn.
  CuMatrix<double> frozen_state;

  OnlineCmvnState() = default;

  explicit OnlineCmvnState(const CuMatrix<double> &global_stats):
      global_cmvn_stats(global_stats) { }

};


} // namespace feat
} // namespace asrlib
} // namespace riva

#endif  // RIVA_ASRLIB_FEAT_ONLINE_FEATURE_H_
