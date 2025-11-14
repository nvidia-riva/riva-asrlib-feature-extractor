// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_C_CU_MATRIX_H_
#define RIVA_ASRLIB_FEAT_C_CU_MATRIX_H_

#include "cu-feat-extr/c-cpp-bridge/cu-container-helper-types.h"
#include "cu-feat-extr/c-cpp-bridge/opaque-types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =======================
// ======= double ========
// =======================
void RivaASRLibFeat_CuMatrixD_t_Create(RivaASRLibFeat_CuMatrixD_t *mat);
void RivaASRLibFeat_CuMatrixD_t_CreateFromOther(RivaASRLibFeat_CuMatrixD_t *mat, const RivaASRLibFeat_CuMatrixD_t *other);
void RivaASRLibFeat_CuMatrixD_t_CopyFromOther(RivaASRLibFeat_CuMatrixD_t *mat, const RivaASRLibFeat_CuMatrixD_t *other);
void RivaASRLibFeat_CuMatrixD_t_CreateWithSize(
  RivaASRLibFeat_CuMatrixD_t *mat,
  RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
  RivaASRLibFeat_ContainerResizeOpt resize_type, RivaASRLibFeat_MatrixStrideOpt stride_type
);

void RivaASRLibFeat_CuMatrixD_t_Destroy(RivaASRLibFeat_CuMatrixD_t *mat);

// other_data_cpu must be row major.
// mat will resize automatically if necessary.
void RivaASRLibFeat_CuMatrixD_t_CopyFromMat(
  RivaASRLibFeat_CuMatrixD_t *mat, const double *other_data_cpu,
  RivaASRLibFeat_ContainerIndex_t other_rows, RivaASRLibFeat_ContainerIndex_t other_cols,
  RivaASRLibFeat_ContainerIndex_t other_stride);

RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixD_t_Rows(RivaASRLibFeat_CuMatrixD_t *mat);
RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixD_t_Cols(RivaASRLibFeat_CuMatrixD_t *mat);
RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixD_t_Stride(RivaASRLibFeat_CuMatrixD_t *mat);

const double* RivaASRLibFeat_CuMatrixD_t_ConstData(const RivaASRLibFeat_CuMatrixD_t *mat);
double* RivaASRLibFeat_CuMatrixD_t_Data(RivaASRLibFeat_CuMatrixD_t *mat);

const double* RivaASRLibFeat_CuMatrixD_t_ConstRowData(const RivaASRLibFeat_CuMatrixD_t *mat, RivaASRLibFeat_ContainerIndex_t row);
double* RivaASRLibFeat_CuMatrixD_t_RowData(RivaASRLibFeat_CuMatrixD_t *mat, RivaASRLibFeat_ContainerIndex_t row);

void RivaASRLibFeat_CuMatrixD_t_Resize(
  RivaASRLibFeat_CuMatrixD_t *mat,
  RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
  RivaASRLibFeat_ContainerResizeOpt resize_type, RivaASRLibFeat_MatrixStrideOpt stride_type);


// =======================
// ======= float ========
// =======================
void RivaASRLibFeat_CuMatrixF_t_Create(RivaASRLibFeat_CuMatrixF_t *mat);
void RivaASRLibFeat_CuMatrixF_t_CreateFromOther(RivaASRLibFeat_CuMatrixF_t *mat, const RivaASRLibFeat_CuMatrixF_t *other);
void RivaASRLibFeat_CuMatrixF_t_CopyFromOther(RivaASRLibFeat_CuMatrixF_t *mat, const RivaASRLibFeat_CuMatrixF_t *other);
void RivaASRLibFeat_CuMatrixF_t_CreateWithSize(
  RivaASRLibFeat_CuMatrixF_t *mat,
  RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
  RivaASRLibFeat_ContainerResizeOpt resize_type, RivaASRLibFeat_MatrixStrideOpt stride_type
);

void RivaASRLibFeat_CuMatrixF_t_Destroy(RivaASRLibFeat_CuMatrixF_t *mat);

// other_data_cpu must be row major.
// mat will resize automatically if necessary.
void RivaASRLibFeat_CuMatrixF_t_CopyFromMat(
  RivaASRLibFeat_CuMatrixF_t *mat, const float *other_data_cpu,
  RivaASRLibFeat_ContainerIndex_t other_rows, RivaASRLibFeat_ContainerIndex_t other_cols,
  RivaASRLibFeat_ContainerIndex_t other_stride);

RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixF_t_Rows(RivaASRLibFeat_CuMatrixF_t *mat);
RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixF_t_Cols(RivaASRLibFeat_CuMatrixF_t *mat);
RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixF_t_Stride(RivaASRLibFeat_CuMatrixF_t *mat);

const float* RivaASRLibFeat_CuMatrixF_t_ConstData(const RivaASRLibFeat_CuMatrixF_t *mat);
float* RivaASRLibFeat_CuMatrixF_t_Data(RivaASRLibFeat_CuMatrixF_t *mat);

const float* RivaASRLibFeat_CuMatrixF_t_ConstRowData(const RivaASRLibFeat_CuMatrixF_t *mat, RivaASRLibFeat_ContainerIndex_t row);
float* RivaASRLibFeat_CuMatrixF_t_RowData(RivaASRLibFeat_CuMatrixF_t *mat, RivaASRLibFeat_ContainerIndex_t row);

void RivaASRLibFeat_CuMatrixF_t_Resize(
  RivaASRLibFeat_CuMatrixF_t *mat,
  RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
  RivaASRLibFeat_ContainerResizeOpt resize_type, RivaASRLibFeat_MatrixStrideOpt stride_type);


#ifdef __cplusplus
}
#endif

#endif