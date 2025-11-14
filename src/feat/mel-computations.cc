// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Phonexia s.r.o.;  Karel Vesely;  Microsoft Corporation
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

#include "cu-feat-extr/feat/feature-window.h"
#include "feat/mel-computations.h"


namespace riva {
namespace asrlib {
namespace feat {

// NOT librosa compatible
static float VtlnWarpFreq(
    float vtln_low_cutoff,
    float vtln_high_cutoff,  // discontinuities in warp func
    float low_freq,
    float high_freq,  // upper+lower frequency cutoffs in
    // the mel computation
    float vtln_warp_factor,
    float freq);

// NOT librosa compatible
static float VtlnWarpMelFreq(
    float vtln_low_cutoff,
    float vtln_high_cutoff,
    float low_freq,
    float high_freq,
    float vtln_warp_factor,
    float mel_freq);

// NOT librosa compatible
static float InverseMelScale_HTK(float mel_freq){
  return 700.0f * (expf (mel_freq / 1127.0f) - 1.0f);
}

// NOT librosa compatible
static float MelScale_HTK(float freq){
  return 1127.0f * logf (1.0f + freq / 700.0f);
}

// librosa compatible
static float InverseMelScale_Stanley(float mel_freq){
  // Use stanley equation
  constexpr float kMinLogHz = 1000.0f;                // Break frequency (Hz)
  constexpr float kStep = 200.0f / 3.0f;              // Step size below break frequency
  constexpr float kMinLogMel = kMinLogHz / kStep;     // Break frequency (Mels)
  /*constexpr*/ float kLogStep = std::log(6.4f) / 27.0f;  // step size above break frequency

  float frequency = mel_freq * kStep;
  // Fill in the log region
  if (mel_freq >= kMinLogMel) {
    frequency = kMinLogHz * std::exp(kLogStep * (mel_freq - kMinLogMel));
  }
  return frequency;
}

// librosa compatible
static float MelScale_Stanley(float freq){
  // Use stanley equation
  constexpr float kMinLogHz = 1000.0f;    // Break frequency (Hz)
  constexpr float kStep = 200.0f / 3.0f;  // Step size below break frequency

  float mel;
  if (freq < kMinLogHz) {
    mel = freq / kStep;
  } else {
    // Fill in the log region
    constexpr float kMinLogMel = kMinLogHz / kStep;     // Break frequency (Mel)
    /*constexpr*/ float kLogStep = std::log(6.4f) / 27.0f;  // step size above break frequency

    mel = kMinLogMel + std::log(freq / kMinLogHz) / kLogStep;
  }
  return mel;
}

// librosa compatible
static std::vector<float> GenerateMelFrequencies(
    int32_t nmels_plus_two, float low_freq, float high_freq) {

  const float min_mel = MelScale_Stanley(low_freq);
  const float max_mel = MelScale_Stanley(high_freq);

  std::vector<float> mel_frequencies(nmels_plus_two);

  for (size_t i = 0; i < mel_frequencies.size(); ++i) {
    float mel_freq = min_mel + 1.0 * i / (mel_frequencies.size() - 1.) * (max_mel - min_mel);
    mel_frequencies[i] = InverseMelScale_Stanley(mel_freq);
  }

  return mel_frequencies;
}

// librosa compatible
static std::vector<float> GenerateFFTFrequencies(float sample_rate, int32_t fft_length){
  std::vector<float> fft_frequencies;
  int num_frequencies = fft_length / 2 + 1;
  fft_frequencies.resize(num_frequencies);
  for (int i = 0; i < num_frequencies; ++i) {
    fft_frequencies[i] = 1.0 * i / (num_frequencies - 1.) * sample_rate / 2.;
  }
  return fft_frequencies;
}


MelBanks::MelBanks(const RivaASRLibFeat_MelBanksOptions &opts,
                   const RivaASRLibFeat_FrameExtractionOptions &frame_opts,
                   float vtln_warp_factor,
                   bool librosa_compat) {
  if(librosa_compat){
    GenerateBins_Librosa(opts, frame_opts);
  }
  else{
    GenerateBins(opts, frame_opts, vtln_warp_factor);
  }
}


void MelBanks::GenerateBins(
    const RivaASRLibFeat_MelBanksOptions &opts,
    const RivaASRLibFeat_FrameExtractionOptions &frame_opts,
    float vtln_warp_factor){

  int32_t num_bins = opts.num_bins;

  if (num_bins < 3) throw std::runtime_error("Must have at least 3 mel bins\n");

  float sample_freq = frame_opts.samp_freq;
  int32_t window_length_padded = frame_opts.padded_window_size;
  assert(window_length_padded % 2 == 0);
  int32_t num_fft_bins = window_length_padded / 2;
  float nyquist = 0.5 * sample_freq;

  float low_freq = opts.low_freq;
  float high_freq;
  if (opts.high_freq > 0.0)
    high_freq = opts.high_freq;
  else
    high_freq = nyquist + opts.high_freq;

  if (low_freq < 0.0 || low_freq >= nyquist
      || high_freq <= 0.0 || high_freq > nyquist
      || high_freq <= low_freq)
    throw std::runtime_error("Bad values in options: low-freq "
              " and high-freq vs. nyquist\n");

  float fft_bin_width = sample_freq / window_length_padded;
  // fft-bin width [think of it as Nyquist-freq / half-window-length]

  float mel_low_freq = MelScale_HTK(low_freq);
  float mel_high_freq = MelScale_HTK(high_freq);

  debug_ = opts.debug_mel;

  // divide by num_bins+1 in next line because of end-effects where the bins
  // spread out to the sides.
  float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins+1);

  float vtln_low = opts.vtln_low,
      vtln_high = opts.vtln_high;
  if (vtln_high < 0.0) {
    vtln_high += nyquist;
  }

  if (vtln_warp_factor != 1.0 &&
      (vtln_low < 0.0 || vtln_low <= low_freq
       || vtln_low >= high_freq
       || vtln_high <= 0.0 || vtln_high >= high_freq
       || vtln_high <= vtln_low))

    throw std::runtime_error("Bad values in options: vtln-low"
                            " and vtln-high versus"
                            " low-freq and high-freq\n");

  bins_.resize(num_bins);

  // NOTE: potential problem: not doing this
  // center_freqs_.Resize(num_bins);

  for (int32_t bin = 0; bin < num_bins; bin++) {
    float left_mel = mel_low_freq + bin * mel_freq_delta,
        center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
        right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

    if (vtln_warp_factor != 1.0) {
      left_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                 vtln_warp_factor, left_mel);
      center_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                 vtln_warp_factor, center_mel);
      right_mel = VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                  vtln_warp_factor, right_mel);
    }

    // center_freqs_(bin) = InverseMelScale_HTK(center_mel);

    // this_bin will be a vector of coefficients that is only
    // nonzero where this mel bin is active.
    std::vector<float> this_bin(num_fft_bins);
    int32_t first_index = -1, last_index = -1;
    for (int32_t i = 0; i < num_fft_bins; i++) {
      float freq = (fft_bin_width * i);  // Center frequency of this fft
                                             // bin.
      float mel = MelScale_HTK(freq);
      if (mel > left_mel && mel < right_mel) {
        float weight;
        if (mel <= center_mel)
          weight = (mel - left_mel) / (center_mel - left_mel);
        else
         weight = (right_mel-mel) / (right_mel-center_mel);
        this_bin[i] = weight;
        if (first_index == -1)
          first_index = i;
        last_index = i;
      }
    }
    assert(first_index != -1 && last_index >= first_index
                 && "You may have set --num-mel-bins too large.");

    bins_[bin].first = first_index;
    int32_t size = last_index + 1 - first_index;

    bins_[bin].second.resize(size);
    bins_[bin].second = std::vector<float>(
      this_bin.begin() + first_index,
      this_bin.begin() + first_index + size);

    // assert(!htk_mode_);
    // Replicate a bug in HTK, for testing purposes.
    // if (opts.htk_mode && bin == 0 && mel_low_freq != 0.0)
    //   bins_[bin].second(0) = 0.0;

  } // for bin idx
}

void MelBanks::GenerateBins_Librosa(
    const RivaASRLibFeat_MelBanksOptions &opts,
    const RivaASRLibFeat_FrameExtractionOptions &frame_opts){

  constexpr bool normalize = true;

  int32_t num_bins = opts.num_bins;
  float sample_freq = frame_opts.samp_freq;
  float nyquist = 0.5 * sample_freq;
  float low_freq = opts.low_freq;
  float high_freq;
  if (opts.high_freq > 0.0)
    high_freq = opts.high_freq;
  else
    high_freq = nyquist + opts.high_freq;

  const auto &fft_length = frame_opts.padded_window_size;
  int num_fft_frequencies = fft_length / 2 + 1;

  const std::vector<float> fft_frequencies = GenerateFFTFrequencies(sample_freq, fft_length);
  const std::vector<float> mel_frequencies = GenerateMelFrequencies(
    num_bins + 2, low_freq, high_freq);

  std::vector<float> delta_frequencies(num_bins + 1);
  for (int bin = 0; bin < num_bins + 1; bin++) {
    delta_frequencies[bin] = mel_frequencies[bin + 1] - mel_frequencies[bin];
  }

  std::vector<std::vector<float>> ramps(num_bins + 2, std::vector<float>(num_fft_frequencies));
  for (int bin = 0; bin < num_bins + 2; bin++) {
    for (int freq = 0; freq < num_fft_frequencies; ++freq) {
      ramps[bin][freq] = (mel_frequencies[bin] - fft_frequencies[freq]);
    }
  }

  bins_.resize(num_bins);
  for (int bin = 0; bin < num_bins; bin++) {
    std::vector<float> non_zero_weights;
    int first_index = -1;
    for (int freq = 0; freq < num_fft_frequencies; freq++) {
      float lower = -ramps[bin][freq] / delta_frequencies[bin];
      float upper = ramps[bin + 2][freq] / delta_frequencies[bin + 1];

      float min = std::min(lower, upper);
      if (min > 0.) {
        non_zero_weights.push_back(min);
        if (first_index < 0)
          first_index = freq;
      }
    }

    bins_[bin].first = first_index;
    bins_[bin].second.resize(non_zero_weights.size());

    std::copy(non_zero_weights.begin(), non_zero_weights.end(), bins_[bin].second.begin());
  }

  if (normalize) {
    std::vector<float> enorm(num_bins);
    for (int bin = 0; bin < num_bins; ++bin) {
      enorm[bin] = 2.0 / (mel_frequencies[bin + 2] - mel_frequencies[bin]);
    }

    for (int bin = 0; bin < num_bins; ++bin) {
      int size = bins_[bin].second.size();
      for (int i = 0; i < size; ++i) {
        bins_[bin].second[i] *= enorm[bin];
      }
    }
  }

}

float VtlnWarpFreq(float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
                                 float vtln_high_cutoff,
                                 float low_freq,  // upper+lower frequency cutoffs in mel computation
                                 float high_freq,
                                 float vtln_warp_factor,
                                 float freq) {
  /// This computes a VTLN warping function that is not the same as HTK's one,
  /// but has similar inputs (this function has the advantage of never producing
  /// empty bins).

  /// This function computes a warp function F(freq), defined between low_freq and
  /// high_freq inclusive, with the following properties:
  ///  F(low_freq) == low_freq
  ///  F(high_freq) == high_freq
  /// The function is continuous and piecewise linear with two inflection
  ///   points.
  /// The lower inflection point (measured in terms of the unwarped
  ///  frequency) is at frequency l, determined as described below.
  /// The higher inflection point is at a frequency h, determined as
  ///   described below.
  /// If l <= f <= h, then F(f) = f/vtln_warp_factor.
  /// If the higher inflection point (measured in terms of the unwarped
  ///   frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
  ///   Since (by the last point) F(h) == h/vtln_warp_factor, then
  ///   max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
  ///   h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
  ///     = vtln_high_cutoff * min(1, vtln_warp_factor).
  /// If the lower inflection point (measured in terms of the unwarped
  ///   frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
  ///   This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
  ///                       = vtln_low_cutoff * max(1, vtln_warp_factor)


  if (freq < low_freq || freq > high_freq) return freq;  // in case this gets called
  // for out-of-range frequencies, just return the freq.

  assert(vtln_low_cutoff > low_freq &&
               "be sure to set the --vtln-low option higher than --low-freq");
  assert(vtln_high_cutoff < high_freq &&
               "be sure to set the --vtln-high option lower than --high-freq [or negative]");
  float one = 1.0;
  float l = vtln_low_cutoff * std::max(one, vtln_warp_factor);
  float h = vtln_high_cutoff * std::min(one, vtln_warp_factor);
  float scale = 1.0 / vtln_warp_factor;
  float Fl = scale * l;  // F(l);
  float Fh = scale * h;  // F(h);
  assert(l > low_freq && h < high_freq);
  // slope of left part of the 3-piece linear function
  float scale_left = (Fl - low_freq) / (l - low_freq);
  // [slope of center part is just "scale"]

  // slope of right part of the 3-piece linear function
  float scale_right = (high_freq - Fh) / (high_freq - h);

  if (freq < l) {
    return low_freq + scale_left * (freq - low_freq);
  } else if (freq < h) {
    return scale * freq;
  } else {  // freq >= h
    return high_freq + scale_right * (freq - high_freq);
  }
}

float VtlnWarpMelFreq(float vtln_low_cutoff,  // upper+lower frequency cutoffs for VTLN.
                                    float vtln_high_cutoff,
                                    float low_freq,  // upper+lower frequency cutoffs in mel computation
                                    float high_freq,
                                    float vtln_warp_factor,
                                    float mel_freq){
  return MelScale_HTK(VtlnWarpFreq(
      vtln_low_cutoff, vtln_high_cutoff,
      low_freq, high_freq,
      vtln_warp_factor,
      InverseMelScale_HTK(mel_freq)));
}


void ComputeLifterCoeffs(float Q, std::vector<float> *coeffs) {
  // Compute liftering coefficients (scaling on cepstral coeffs)
  // coeffs are numbered slightly differently from HTK: the zeroth
  // index is C0, which is not affected.
  for (int32_t i = 0; i < coeffs->size(); i++)
    (*coeffs)[i] = 1.0 + 0.5 * Q * sin (M_PI * i / Q);
}


} // namespace feat
} // namespace asrlib
} // namespace riva
