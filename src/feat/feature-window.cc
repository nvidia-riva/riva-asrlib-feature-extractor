// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation
// SPDX-FileCopyrightText: Copyright (c) 2013-2016  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2014       IMSL, PKU-HKUST (author: Wei Shi)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/feat/feature-window.h"
#include <stdint.h>

#include <stdexcept>
#include <cassert>

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

namespace riva {
namespace asrlib {
namespace feat {

int64_t FirstSampleOfFrame(
  int32_t frame,
  const RivaASRLibFeat_FrameExtractionOptions &opts)
{

  int64_t frame_shift = opts.window_shift;

  if(opts.frame_rnd_strat == RivaASRLibFeat_FrameRoundingStrategy_Nearest){
    int64_t midpoint_of_frame = frame_shift * frame  +  frame_shift / 2;
    int64_t beginning_of_frame = midpoint_of_frame  -  opts.window_size / 2;
    return beginning_of_frame;
  }

  if(opts.frame_rnd_strat == RivaASRLibFeat_FrameRoundingStrategy_SnipEdges
      || opts.frame_rnd_strat == RivaASRLibFeat_FrameRoundingStrategy_DownAddOne){
    return frame * frame_shift;
  }

  printf("\nWARNING: Unhandled rounding strategy in FirstSampleOfFrame!\n");
  assert(false);
  return 0;
}

int32_t NumFrames(int64_t num_samples,
                const RivaASRLibFeat_FrameExtractionOptions &opts,
                bool flush) {

  int64_t frame_shift = opts.window_shift;
  int64_t frame_length = opts.window_size;

  if(opts.frame_rnd_strat == RivaASRLibFeat_FrameRoundingStrategy_DownAddOne){
    // WARNING: This rounding has not been tested when library is run in online mode.
    if(num_samples == 0)
      return 0;
    else
      return  num_samples / frame_shift + 1;
  }
  else if(opts.frame_rnd_strat == RivaASRLibFeat_FrameRoundingStrategy_SnipEdges){
    // with --snip-edges=true (the default), we use a HTK-like approach to
    // determining the number of frames-- all frames have to fit completely into
    // the waveform, and the first frame begins at sample zero.
    if (num_samples < frame_length)
      return 0;
    else
      return (1 + ((num_samples - frame_length) / frame_shift));
    // You can understand the expression above as follows: 'num_samples -
    // frame_length' is how much room we have to shift the frame within the
    // waveform; 'frame_shift' is how much we shift it each time; and the ratio
    // is how many times we can shift it (integer arithmetic rounds down).
  }
  else if(opts.frame_rnd_strat == RivaASRLibFeat_FrameRoundingStrategy_Nearest){
    // if --snip-edges=false, the number of frames is determined by rounding the
    // (file-length / frame-shift) to the nearest integer.  The point of this
    // formula is to make the number of frames an obvious and predictable
    // function of the frame shift and signal length, which makes many
    // segmentation-related questions simpler.
    //
    // Because integer division in C++ rounds toward zero, we add (half the
    // frame-shift minus epsilon) before dividing, to have the effect of
    // rounding towards the closest integer.
    int32_t num_frames = (num_samples + (frame_shift / 2)) / frame_shift;

    if (flush)
      return num_frames;

    // note: 'end' always means the last plus one, i.e. one past the last.
    int64_t end_sample_of_last_frame = FirstSampleOfFrame(num_frames - 1, opts)
        + frame_length;

    // the following code is optimized more for clarity than efficiency.
    // If flush == false, we can't output frames that extend past the end
    // of the signal.
    while (num_frames > 0 && end_sample_of_last_frame > num_samples) {
      num_frames--;
      end_sample_of_last_frame -= frame_shift;
    }
    return num_frames;
  }
  else {
    printf("\nWARNING: Unhandled rounding strategy inside NumFrames\n");
    assert(false);
    return 0;
  }
}


FeatureWindowFunction::FeatureWindowFunction(const RivaASRLibFeat_FrameExtractionOptions &opts) {
  int32_t frame_length = opts.window_size;
  assert(frame_length > 0);
  window.resize(frame_length);
  double a = M_2PI / (frame_length-1);
  for (int32_t i = 0; i < frame_length; i++) {
    double i_fl = static_cast<double>(i);
    switch(opts.window_type){
      case RivaASRLibFeat_HANNING:
        window[i] = 0.5  - 0.5*cos(a * i_fl);
        break;
      case RivaASRLibFeat_SINE:
        // when you are checking ws wikipedia, please
        // note that 0.5 * a = M_PI/(frame_length-1)
        window[i] = sin(0.5 * a * i_fl);
        break;
      case RivaASRLibFeat_HAMMING:
        window[i] = 0.54 - 0.46*cos(a * i_fl);
        break;
      case RivaASRLibFeat_POVEY:
        // like hamming but goes to zero at edges.
        window[i] = pow(0.5 - 0.5*cos(a * i_fl), 0.85);
        break;
      case RivaASRLibFeat_RECTANGULAR:
        window[i] = 1.0;
        break;
      case RivaASRLibFeat_BLACKMAN:
        window[i] = opts.blackman_coeff - 0.5*cos(a * i_fl) +
          (0.5 - opts.blackman_coeff) * cos(2 * a * i_fl);
        break;
      default:
        throw std::runtime_error(std::string("Invalid window type\n"));
    }
  }
}


} // namespace feat
} // namespace asrlib
} // namespace riva
