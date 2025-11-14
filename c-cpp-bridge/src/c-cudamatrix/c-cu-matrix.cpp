// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/c-cpp-bridge/c-cudamatrix/c-cu-matrix.h"

#include "cu-feat-extr/cudamatrix/cu-matrix.h"

extern "C" {

using riva::asrlib::feat::CuMatrix;

// opaque casting is also done in online-batched-feature-info.h::GetGlobalCmvnStats()
// Want a shortcut to avoid duplication of reinterpret_cast<...>
// Not functions for sake of conveniently having an lvalue, without needing another *

// Given a CCuMatrixX_t*, provides a CuMatrix<X>*, available as an lvalue.
#define REINTERPRET_mat_d(mat) (*reinterpret_cast< CuMatrix<double>** > ((mat)))
#define REINTERPRET_mat_d_const(mat) (*reinterpret_cast< CuMatrix<double> * const * > ((mat)))

// Given a const CCuMatrixX_t*, provides a CuMatrix<X>* const
#define REINTERPRET_mat_f(mat) (*reinterpret_cast< CuMatrix<float>** > ((mat)))
#define REINTERPRET_mat_f_const(mat) (*reinterpret_cast< CuMatrix<float> * const * > ((mat)))


// =======================
// ======= double ========
// =======================

void RivaASRLibFeat_CuMatrixD_t_Create(RivaASRLibFeat_CuMatrixD_t *mat){
    REINTERPRET_mat_d(mat) = new CuMatrix<double>;
}
void RivaASRLibFeat_CuMatrixD_t_CreateFromOther(RivaASRLibFeat_CuMatrixD_t *mat, const RivaASRLibFeat_CuMatrixD_t *other){
    RivaASRLibFeat_CuMatrixD_t_Create(mat);
    RivaASRLibFeat_CuMatrixD_t_CopyFromOther(mat, other);
}
void RivaASRLibFeat_CuMatrixD_t_CopyFromOther(RivaASRLibFeat_CuMatrixD_t *mat, const RivaASRLibFeat_CuMatrixD_t *other){
    *REINTERPRET_mat_d(mat) = *REINTERPRET_mat_d_const(other);
}

void RivaASRLibFeat_CuMatrixD_t_CreateWithSize(
        RivaASRLibFeat_CuMatrixD_t *mat,
        RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
        RivaASRLibFeat_ContainerResizeOpt resize_type, RivaASRLibFeat_MatrixStrideOpt stride_type){
    REINTERPRET_mat_d(mat) = new CuMatrix<double>(rows, cols, resize_type, stride_type);
}

void RivaASRLibFeat_CuMatrixD_t_Destroy(RivaASRLibFeat_CuMatrixD_t *mat){
    delete REINTERPRET_mat_d(mat);
    REINTERPRET_mat_d(mat) = nullptr;
}

void RivaASRLibFeat_CuMatrixD_t_CopyFromMat(
    RivaASRLibFeat_CuMatrixD_t *mat, const double *other_data_cpu,
    RivaASRLibFeat_ContainerIndex_t other_rows, RivaASRLibFeat_ContainerIndex_t other_cols,
    RivaASRLibFeat_ContainerIndex_t other_stride){

    REINTERPRET_mat_d(mat)->CopyFromMat(other_data_cpu,
        other_rows, other_cols, other_stride);
}

RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixD_t_Rows(RivaASRLibFeat_CuMatrixD_t *mat){
    return REINTERPRET_mat_d(mat)->NumRows();
}
RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixD_t_Cols(RivaASRLibFeat_CuMatrixD_t *mat){
    return REINTERPRET_mat_d(mat)->NumCols();
}
RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixD_t_Stride(RivaASRLibFeat_CuMatrixD_t *mat){
    return REINTERPRET_mat_d(mat)->Stride();
}

const double* RivaASRLibFeat_CuMatrixD_t_ConstData(const RivaASRLibFeat_CuMatrixD_t *mat){
    return REINTERPRET_mat_d_const(mat)->Data();
}
double* RivaASRLibFeat_CuMatrixD_t_Data(RivaASRLibFeat_CuMatrixD_t *mat){
    return REINTERPRET_mat_d(mat)->Data();
}

const double* RivaASRLibFeat_CuMatrixD_t_ConstRowData(const RivaASRLibFeat_CuMatrixD_t *mat, RivaASRLibFeat_ContainerIndex_t row){
    return REINTERPRET_mat_d_const(mat)->RowData(row);
}
double* RivaASRLibFeat_CuMatrixD_t_RowData(RivaASRLibFeat_CuMatrixD_t *mat, RivaASRLibFeat_ContainerIndex_t row){
    return REINTERPRET_mat_d(mat)->RowData(row);
}

void RivaASRLibFeat_CuMatrixD_t_Resize(
        RivaASRLibFeat_CuMatrixD_t *mat,
        RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
        RivaASRLibFeat_ContainerResizeOpt resize_type, RivaASRLibFeat_MatrixStrideOpt stride_type){

    REINTERPRET_mat_d(mat)->Resize(rows, cols, resize_type, stride_type);
}


// =======================
// ======= float ========
// =======================
void RivaASRLibFeat_CuMatrixF_t_Create(RivaASRLibFeat_CuMatrixF_t *mat){
    REINTERPRET_mat_f(mat) = new CuMatrix<float>;
}

void RivaASRLibFeat_CuMatrixF_t_CreateFromOther(RivaASRLibFeat_CuMatrixF_t *mat, const RivaASRLibFeat_CuMatrixF_t *other){
    RivaASRLibFeat_CuMatrixF_t_Create(mat);
    RivaASRLibFeat_CuMatrixF_t_CopyFromOther(mat, other);
}
void RivaASRLibFeat_CuMatrixF_t_CopyFromOther(RivaASRLibFeat_CuMatrixF_t *mat, const RivaASRLibFeat_CuMatrixF_t *other){
    *REINTERPRET_mat_f(mat) = *REINTERPRET_mat_f_const(other);
}

void RivaASRLibFeat_CuMatrixF_t_CreateWithSize(
        RivaASRLibFeat_CuMatrixF_t *mat,
        RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
        RivaASRLibFeat_ContainerResizeOpt resize_type, RivaASRLibFeat_MatrixStrideOpt stride_type){

    REINTERPRET_mat_f(mat) = new CuMatrix<float>(rows, cols, resize_type, stride_type);
}

void RivaASRLibFeat_CuMatrixF_t_Destroy(RivaASRLibFeat_CuMatrixF_t *mat){
    delete REINTERPRET_mat_f(mat);
    REINTERPRET_mat_f(mat) = nullptr;
}

void RivaASRLibFeat_CuMatrixF_t_CopyFromMat(
        RivaASRLibFeat_CuMatrixF_t *mat, const float *other_data_cpu,
        RivaASRLibFeat_ContainerIndex_t other_rows, RivaASRLibFeat_ContainerIndex_t other_cols,
        RivaASRLibFeat_ContainerIndex_t other_stride){
    REINTERPRET_mat_f(mat)->CopyFromMat(other_data_cpu,
        other_rows, other_cols, other_stride);
}

RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixF_t_Rows(RivaASRLibFeat_CuMatrixF_t *mat){
    return REINTERPRET_mat_f(mat)->NumRows();
}
RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixF_t_Cols(RivaASRLibFeat_CuMatrixF_t *mat){
    return REINTERPRET_mat_f(mat)->NumCols();
}
RivaASRLibFeat_ContainerIndex_t RivaASRLibFeat_CuMatrixF_t_Stride(RivaASRLibFeat_CuMatrixF_t *mat){
    return REINTERPRET_mat_f(mat)->Stride();
}

const float* RivaASRLibFeat_CuMatrixF_t_ConstData(const RivaASRLibFeat_CuMatrixF_t *mat){
    return REINTERPRET_mat_f_const(mat)->Data();
}
float* RivaASRLibFeat_CuMatrixF_t_Data(RivaASRLibFeat_CuMatrixF_t *mat){
    return REINTERPRET_mat_f(mat)->Data();
}

const float* RivaASRLibFeat_CuMatrixF_t_ConstRowData(const RivaASRLibFeat_CuMatrixF_t *mat, RivaASRLibFeat_ContainerIndex_t row){
    return REINTERPRET_mat_f_const(mat)->RowData(row);
}
float* RivaASRLibFeat_CuMatrixF_t_RowData(RivaASRLibFeat_CuMatrixF_t *mat, RivaASRLibFeat_ContainerIndex_t row){
    return REINTERPRET_mat_f(mat)->RowData(row);
}

void RivaASRLibFeat_CuMatrixF_t_Resize(
        RivaASRLibFeat_CuMatrixF_t *mat,
        RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
        RivaASRLibFeat_ContainerResizeOpt resize_type, RivaASRLibFeat_MatrixStrideOpt stride_type){

    REINTERPRET_mat_f(mat)->Resize(rows, cols, resize_type, stride_type);
}


} // extern c