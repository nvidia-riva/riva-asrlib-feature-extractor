// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
// SPDX-FileCopyrightText: Copyright (c) 2014-2016  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FEATURE_WINDOW_H_
#define RIVA_ASRLIB_FEAT_FEATURE_WINDOW_H_

#include <map>
#include <string>
#include <vector>

#include <stdint.h>
#include "cu-feat-extr/cudamatrix/cu-vector.h"
#include "cu-feat-extr/cudamatrix/cu-common.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"
#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"

namespace riva {
namespace asrlib {
namespace feat {

struct FeatureWindowFunction {
  FeatureWindowFunction() {}
  explicit FeatureWindowFunction(const RivaASRLibFeat_FrameExtractionOptions &opts);
  FeatureWindowFunction(const FeatureWindowFunction &other):
      window(other.window) { }
  std::vector<float> window;
};


/**
   This function returns the number of frames that we can extract from a wave
   file with the given number of samples in it (assumed to have the same
   sampling rate as specified in 'opts').

      @param [in] num_samples  The number of samples in the wave file.
      @param [in] opts     The frame-extraction options class

      @param [in] flush   True if we are asserting that this number of samples is
             'all there is', false if we expecting more data to possibly come
             in.  This only makes a difference to the answer if opts.snips_edges
             == false.  For offline feature extraction you always want flush ==
             true.  In an online-decoding context, once you know (or decide) that
             no more data is coming in, you'd call it with flush == true at the
             end to flush out any remaining data.
*/
int32_t NumFrames(int64_t num_samples,
                const RivaASRLibFeat_FrameExtractionOptions &opts,
                bool flush = true);

/*
   This function returns the index of the first sample of the frame indexed
   'frame'.  If snip-edges=true, it just returns frame * opts.window_shift; if
   snip-edges=false, the formula is a little more complicated and the result may
   be negative.
*/
int64_t FirstSampleOfFrame(int32_t frame,
                         const RivaASRLibFeat_FrameExtractionOptions &opts);


} // namespace feat
} // namespace asrlib
} // namespace riva


#endif  // RIVA_ASRLIB_FEAT_FEATURE_WINDOW_H_
