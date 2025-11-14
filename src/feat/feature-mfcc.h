// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
// SPDX-FileCopyrightText: Copyright (c) 2014-2016  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FEATURE_MFCC_H_
#define RIVA_ASRLIB_FEAT_FEATURE_MFCC_H_

#include <map>
#include <string>
#include <vector>
#include <stdint.h>
#include <stdexcept>

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-mfcc-opts.h"
#include "cu-feat-extr/feat/feature-window.h"
#include "cu-feat-extr/cudamatrix/cu-matrix.h"
#include "cu-feat-extr/cudamatrix/cu-vector.h"
#include "feat/mel-computations.h"

namespace riva {
namespace asrlib {
namespace feat {

class CudaSpectralFeatureOptions;

// This is the new-style interface to the MFCC computation.
class MfccComputer {
 public:
  explicit MfccComputer(const CudaSpectralFeatureOptions &cumfcc_opts);

  const RivaASRLibFeat_FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  int32_t Dim() const { return opts_.num_ceps; }

  bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }


  ~MfccComputer() {
    for (std::map<float, MelBanks*>::iterator iter = mel_banks_.begin();
        iter != mel_banks_.end();
        ++iter)
      delete iter->second;
  //   delete srfft_;
  }

  MfccComputer(const MfccComputer &other) = delete;
  MfccComputer& operator=(const MfccComputer &in) = delete;

 protected:
  const MelBanks *GetMelBanks(float vtln_warp, bool librosa_compat) {
    MelBanks *this_mel_banks = NULL;
    std::map<float, MelBanks*>::iterator iter = mel_banks_.find(vtln_warp);
    if (iter == mel_banks_.end()) {
      this_mel_banks = new MelBanks(opts_.mel_opts,
                                    opts_.frame_opts,
                                    vtln_warp,
                                    librosa_compat);
      mel_banks_[vtln_warp] = this_mel_banks;
    } else {
      this_mel_banks = iter->second;
    }
    return this_mel_banks;
  }

  RivaASRLibFeat_MfccOptions opts_;
  std::vector<float> lifter_coeffs_;
  CuMatrix<float> dct_matrix_;  // matrix we left-multiply by to perform DCT.
//   float log_energy_floor_;
  std::map<float, MelBanks*> mel_banks_;  // float is VTLN coefficient.
//   SplitRadixRealFft<float> *srfft_;

//   // note: mel_energies_ is specific to the frame we're processing, it's
//   // just a temporary workspace.
//   Vector<float> mel_energies_;
};

} // namespace feat
} // namespace asrlib
} // namespace riva


#endif  // RIVA_ASRLIB_FEAT_FEATURE_MFCC_H_
