// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University.
// SPDX-FileCopyrightText: Copyright (c) 2014-2016  Johns Hopkins University (author: Daniel Povey).
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_C_FRAME_EXTRACTION_OPTS_H_
#define RIVA_ASRLIB_FEAT_C_FRAME_EXTRACTION_OPTS_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  RivaASRLibFeat_HAMMING,
  RivaASRLibFeat_RECTANGULAR,
  RivaASRLibFeat_POVEY,
  RivaASRLibFeat_HANNING,
  RivaASRLibFeat_SINE,
  RivaASRLibFeat_BLACKMAN
} RivaASRLibFeat_WindowType;

typedef enum {
  RivaASRLibFeat_FrameRoundingStrategy_SnipEdges,
  RivaASRLibFeat_FrameRoundingStrategy_Nearest,

  // NOTE: WARNING: DownAddOne has never been tested when library is run in online mode.
  // ie when NOT all calls always have first and last = true when identifying chunks.
  RivaASRLibFeat_FrameRoundingStrategy_DownAddOne,

} RivaASRLibFeat_FrameRoundingStrategy;

/*
NOTE:
If you change change samp_freq, frame_shift_ms, or frame_length_ms after construction, call recompute function.
  window_shift, window_size, and padded_window_size
  were getter functions in the past. Now these are computed once.
  Would prefer not to have functions associated with this struct so that
  the C++ code does not need to generate a FrameExtractionOptions when it only has a RivaASRLibFeat_FrameExtractionOptions.
*/
typedef struct {

  float samp_freq;

  // in milliseconds.
  float frame_shift_ms;

  // in milliseconds.
  float frame_length_ms;

  // Amount of dithering, 0.0 means no dither.
  float dither;

  // Preemphasis coefficient.
  float preemph_coeff;

  // Subtract mean of wave before FFT.
  bool remove_dc_offset;

  // e.g. Hamming window
  // May be RivaASRLibFeat_(HAMMING|RECTANGULAR|POVEY|HANNING|SINE|BLACKMAN)
  // "povey" is a window similar to Hamming but goes to zero at the edges.
  // It's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85)
  RivaASRLibFeat_WindowType window_type;

  // NOTE: This replaces the old member variable "snip_edges".
  // For old snip_edges = true, select SnipEdges
  // For old snip_edges = false, select Nearest
  RivaASRLibFeat_FrameRoundingStrategy frame_rnd_strat;

  bool round_to_power_of_two;
  float blackman_coeff;

  /*
  Say we need to reflect the wav during frame extraction, where ABC is the data that we have
  and we need to reflect over the left side / beginning of the signal.
  true  results in:   ...cb | ABC...  (reflects as if the element is the mirror). (default)
  false results in:  ...cba | ABC...  (reflects as if the data boundary is the mirror). (riva/librosa)
  */
  bool reflect_over_element;

  /*
  The "center" parameter from Librosa's librosa.stft().
  NOTE: When this option is enabled, the library will automatically increase
    the value of max_chunk_size_samples given to OnlineBatchedFeaturePipelineCuda's ctor by
    fft_length aka padded_window_size.
    The library will virtually add the 0.5*fft_length worth of padding on each side of the signal.
    (User need not add this padding themselves.)
  When window size is rounded up to power of 2 for padding fft, this parameter defines where the padding goes.
  false -> All padding at end/tail.
  true -> Equal padding on left and right ends.
  */
  bool librosa_center;

  // DERIVED FROM ABOVE
  int32_t window_shift;
  int32_t window_size;
  int32_t padded_window_size;

} RivaASRLibFeat_FrameExtractionOptions;

// Provides default values
void RivaASRLibFeat_FrameExtractionOptions_Create(RivaASRLibFeat_FrameExtractionOptions *opts);
void RivaASRLibFeat_FrameExtractionOptions_Destroy(RivaASRLibFeat_FrameExtractionOptions *opts);

// TODO: Make it so users do not have to worry about calling this.
void RivaASRLibFeat_FrameExtractionOptions_RecomputeWindowParams(RivaASRLibFeat_FrameExtractionOptions *opts);

void RivaASRLibFeat_FrameExtractionOptions_Print(const RivaASRLibFeat_FrameExtractionOptions *opts);

#ifdef __cplusplus
}
#endif

#endif
