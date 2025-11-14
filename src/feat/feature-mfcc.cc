// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek
// SPDX-FileCopyrightText: Copyright (c) 2016       Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// This file exists to avoid mutual header dependency feature-mfcc.h <> feature-onilne-batched-spectral-cuda.h

#include "feat/feature-mfcc.h"

#include "cudafeat/feature-online-batched-spectral-cuda.h"

namespace riva {
namespace asrlib {
namespace feat {


MfccComputer::MfccComputer(const CudaSpectralFeatureOptions &cumfcc_opts):
        opts_(cumfcc_opts.mfcc_opts) /*, srfft_(NULL),
        mel_energies_(opts.mel_opts.num_bins) */ {

    const RivaASRLibFeat_MfccOptions &opts = cumfcc_opts.mfcc_opts;
    int32_t num_bins = opts.mel_opts.num_bins;
    if (opts.num_ceps > num_bins){
        throw std::runtime_error( "num-ceps cannot be larger than num-mel-bins.\n");
    }

    // Matrix<float> dct_matrix(num_bins, num_bins);
    // ComputeDctMatrix(&dct_matrix);
    // // Note that we include zeroth dct in either case.  If using the
    // // energy we replace this with the energy.  This means a different
    // // ordering of features than HTK.
    // SubMatrix<float> dct_rows(dct_matrix, 0, opts.num_ceps, 0, num_bins);

    dct_matrix_.Resize(opts.num_ceps, num_bins);

    dct_matrix_.ComputeDctMatrix();
    // dct_matrix_.CopyFromMat(dct_rows);  // subset of rows.

    if (opts.cepstral_lifter != 0.0) {
        lifter_coeffs_.resize(opts.num_ceps);
        ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);
    }
    // if (opts.energy_floor > 0.0)
    //   log_energy_floor_ = Log(opts.energy_floor);

    // int32_t padded_window_size = opts.frame_opts.padded_window_size;
    // if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    //   srfft_ = new SplitRadixRealFft<float>(padded_window_size);

    // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
    // [note: this call caches it.]
    // NOTE: this librosa_compat is not initialized correctly.
    GetMelBanks(1.0, cumfcc_opts.librosa_compat);

}

} // namespace feat
} // namespace asrlib
} // namespace riva