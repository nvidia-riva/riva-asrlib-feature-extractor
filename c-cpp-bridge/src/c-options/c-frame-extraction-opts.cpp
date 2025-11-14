// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
// SPDX-FileCopyrightText: Copyright (c) 2014-2016  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/c-cpp-bridge/structs/c-options/c-frame-extraction-opts.h"

#include <cassert>
#include <string>
#include <cstdio>

extern "C" {

static int32_t RoundUpToNearestPowerOfTwo(int32_t n);
static std::string window_type_enum_to_str(RivaASRLibFeat_WindowType window_type);
static std::string frame_rnd_strat_enum_to_str(RivaASRLibFeat_FrameRoundingStrategy frs);

void RivaASRLibFeat_FrameExtractionOptions_Create(RivaASRLibFeat_FrameExtractionOptions *opts){

    opts->samp_freq = 16000;
    opts->frame_shift_ms = 10.0;
    opts->frame_length_ms = 25.0;
    opts->dither = 1.0;
    opts->preemph_coeff = 0.97;
    opts->remove_dc_offset = true;
    opts->window_type = RivaASRLibFeat_POVEY;
    opts->round_to_power_of_two = true;
    opts->blackman_coeff = 0.42;
    opts->reflect_over_element = true;
    opts->frame_rnd_strat = RivaASRLibFeat_FrameRoundingStrategy_SnipEdges;
    opts->librosa_center = false;

    RivaASRLibFeat_FrameExtractionOptions_RecomputeWindowParams(opts);
}

void RivaASRLibFeat_FrameExtractionOptions_Destroy(RivaASRLibFeat_FrameExtractionOptions *opts){
    // nothing for now.
}

void RivaASRLibFeat_FrameExtractionOptions_RecomputeWindowParams(RivaASRLibFeat_FrameExtractionOptions *opts){

    // Order of these matters
    opts->window_shift = static_cast<int32_t>(
        opts->samp_freq * 0.001 * opts->frame_shift_ms
    );

    opts->window_size = static_cast<int32_t>(
        opts->samp_freq * 0.001 * opts->frame_length_ms
    );

    opts->padded_window_size = opts->round_to_power_of_two ?
        RoundUpToNearestPowerOfTwo(opts->window_size)
        : opts->window_size;
}

void RivaASRLibFeat_FrameExtractionOptions_Print(const RivaASRLibFeat_FrameExtractionOptions *opts){

    printf("Frame Extraction Options:\n");
    printf("samp_freq: %f\n", opts->samp_freq);
    printf("frame_shift_ms: %f\n", opts->frame_shift_ms);
    printf("frame_length_ms: %f\n", opts->frame_length_ms);
    printf("dither: %f\n", opts->dither);
    printf("preemph_coeff: %f\n", opts->preemph_coeff);
    printf("remove_dc_offset: %d\n", opts->remove_dc_offset);
    printf("window_type: %s\n", window_type_enum_to_str(opts->window_type).c_str());
    printf("round_to_power_of_two: %d\n", opts->round_to_power_of_two);
    printf("blackman_coeff: %f\n", opts->blackman_coeff);
    printf("reflect_over_element: %d\n", opts->reflect_over_element);
    printf("frame_rounding_strategy: %s\n", frame_rnd_strat_enum_to_str(opts->frame_rnd_strat).c_str());
    printf("window_shift: %i\n", opts->window_shift);
    printf("window_size: %i\n", opts->window_size);
    printf("padded_window_size: %i\n", opts->padded_window_size);
    printf("librosa_center: %d\n", opts->librosa_center);
    printf("\n");
}

int32_t RoundUpToNearestPowerOfTwo(int32_t n) {
    assert(n > 0);
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

std::string window_type_enum_to_str(RivaASRLibFeat_WindowType window_type){

    switch(window_type){
        case RivaASRLibFeat_HANNING:     return "hanning";
        case RivaASRLibFeat_SINE:        return "sine";
        case RivaASRLibFeat_HAMMING:     return "hamming";
        case RivaASRLibFeat_POVEY:       return "povey";
        case RivaASRLibFeat_RECTANGULAR: return "rectangular";
        case RivaASRLibFeat_BLACKMAN:    return "blackman";
        default:
            assert(false);
            return "invalid";
    }
}

std::string frame_rnd_strat_enum_to_str(RivaASRLibFeat_FrameRoundingStrategy frs){

    switch(frs){
        case RivaASRLibFeat_FrameRoundingStrategy_SnipEdges:  return "snip_edges";
        case RivaASRLibFeat_FrameRoundingStrategy_Nearest:    return "nearest";
        case RivaASRLibFeat_FrameRoundingStrategy_DownAddOne: return "down_add_one";
        default:
            assert(false);
            return "invalid";
    }
}

} // extern c