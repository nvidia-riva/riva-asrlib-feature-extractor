// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_OPAQUE_TYPES_H_
#define RIVA_ASRLIB_FEAT_OPAQUE_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

// forward declare
// These exact 3 symbols are not to be seen outside this header.
struct RivaASRLibFeat_OnlineBatchedFeaturePipelineCuda;
struct RivaASRLibFeat_CuMatrixD; // in reality, a CuMatrix<double>
struct RivaASRLibFeat_CuMatrixF; // in reality, a CuMatrix<float>

// alias
typedef struct RivaASRLibFeat_OnlineBatchedFeaturePipelineCuda* RivaASRLibFeat_OnlineBatchedFeaturePipelineCuda_t;
typedef struct RivaASRLibFeat_CuMatrixD* RivaASRLibFeat_CuMatrixD_t;
typedef struct RivaASRLibFeat_CuMatrixF* RivaASRLibFeat_CuMatrixF_t;

#ifdef __cplusplus
}
#endif

#endif