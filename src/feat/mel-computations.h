// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Phonexia s.r.o.;  Microsoft Corporation
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_MEL_COMPUTATIONS_H_
#define RIVA_ASRLIB_FEAT_MEL_COMPUTATIONS_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <utility>
#include <vector>

#include <stdint.h>
#include "cu-feat-extr/cudamatrix/cu-vector.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mel-banks-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

class MelBanks {
 public:

  MelBanks(const RivaASRLibFeat_MelBanksOptions &opts,
           const RivaASRLibFeat_FrameExtractionOptions &frame_opts,
           float vtln_warp_factor,
           bool librosa_compat);

  int32_t NumBins() const { return bins_.size(); }

  const std::vector<std::pair<int32_t, std::vector<float> > >& GetBins() const {
    return bins_;
  }

  MelBanks(const MelBanks &other) = default;

  MelBanks &operator = (const MelBanks &other) = delete;

 private:

  void GenerateBins(const RivaASRLibFeat_MelBanksOptions &opts,
                   const RivaASRLibFeat_FrameExtractionOptions &frame_opts,
                   float vtln_warp_factor);

  void GenerateBins_Librosa(const RivaASRLibFeat_MelBanksOptions &opts,
                   const RivaASRLibFeat_FrameExtractionOptions &frame_opts);

  // center frequencies of bins, numbered from 0 ... num_bins-1.
  // Needed by GetCenterFreqs().
  // NOTE: potential problem: not doing this.
  // CuVector<float> center_freqs_;

  // the "bins_" vector is a vector, one for each bin, of a pair:
  // (the first nonzero fft-bin), (the vector of weights).
  std::vector<std::pair<int32_t, std::vector<float> > > bins_;

  bool debug_;
};


// Compute liftering coefficients (scaling on cepstral coeffs)
// coeffs are numbered slightly differently from HTK: the zeroth
// index is C0, which is not affected.
void ComputeLifterCoeffs(float Q, std::vector<float> *coeffs);

} // namespace feat
} // namespace asrlib
} // namespace riva

#endif  // RIVA_ASRLIB_FEAT_MEL_COMPUTATIONS_H_
