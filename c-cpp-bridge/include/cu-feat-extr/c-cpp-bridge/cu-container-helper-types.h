// SPDX-FileCopyrightText: Copyright (c) 2009-2011  Microsoft Corporation.
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_CU_CONTAINER_HELPER_TYPES_H_
#define RIVA_ASRLIB_FEAT_CU_CONTAINER_HELPER_TYPES_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif


typedef uint32_t RivaASRLibFeat_ContainerIndex_t;
typedef enum {
  RivaASRLibFeat_kSetZero,
  RivaASRLibFeat_kUndefined,
  RivaASRLibFeat_kCopyData
} RivaASRLibFeat_ContainerResizeOpt;

typedef enum {
  RivaASRLibFeat_kDefaultStride,
  RivaASRLibFeat_kStrideEqualNumCols,
} RivaASRLibFeat_MatrixStrideOpt;

typedef enum {
  RivaASRLibFeat_kTrans = 112,   // = CblasTrans
  RivaASRLibFeat_kNoTrans = 111  // = CblasNoTrans
} RivaASRLibFeat_MatrixTransposeOpt;


#ifdef __cplusplus
}
#endif

#endif