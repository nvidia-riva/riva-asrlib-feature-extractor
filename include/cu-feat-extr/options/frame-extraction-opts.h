// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
// SPDX-FileCopyrightText: Copyright (c) 2014-2016  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_FRAME_EXTRACTION_OPTS_H_
#define RIVA_ASRLIB_FEAT_FRAME_EXTRACTION_OPTS_H_

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"
#include <cassert>

namespace riva {
namespace asrlib {
namespace feat {

struct FrameExtractionOptions {

  RivaASRLibFeat_FrameExtractionOptions c_opts;

  FrameExtractionOptions(){
    RivaASRLibFeat_FrameExtractionOptions_Create(&c_opts);
  }
  ~FrameExtractionOptions(){
    RivaASRLibFeat_FrameExtractionOptions_Destroy(&c_opts);
  }

  FrameExtractionOptions(const FrameExtractionOptions &other) = default;
  FrameExtractionOptions(FrameExtractionOptions &&other) = default;

  FrameExtractionOptions& operator=(const FrameExtractionOptions &other) = default;
  FrameExtractionOptions& operator=(FrameExtractionOptions &&other) = default;

};

} // namespace feat
} // namespace asrlib
} // namespace riva

#endif
