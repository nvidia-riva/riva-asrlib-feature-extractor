// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely, Lucas Ondel
// SPDX-FileCopyrightText: Copyright (c)      2013  Ehsan Variani, Karel Vesely, Johns Hopkins University (author: Daniel Povey), Hainan Xu, Xiaohui Zhang
// SPDX-FileCopyrightText: Copyright (c) 2013-2015  Guoguo Chen
// SPDX-FileCopyrightText: Copyright (c)      2015  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c) 2016-2017  Shiyin Kang
// SPDX-FileCopyrightText: Copyright (c)      2017  Hossein Hadian
// SPDX-FileCopyrightText: Copyright (c)      2019  Yiwen Shao
// SPDX-FileCopyrightText: Copyright (c)      2025  NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/cudamatrix/cu-matrix.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <utility> // std::swap

#include <cublas_v2.h>

#include <stdint.h>
#include "cu-feat-extr/cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"

#include "assert.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

typedef uint32_t         uint32_cuda;
typedef int32_t          int32_cuda;

namespace riva {
namespace asrlib {
namespace feat {

inline cublasStatus_t
cublas_gemm(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
  return cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline cublasStatus_t
cublas_gemm(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C,
    int ldc)
{
  return cublasDgemm_v2(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

static void GetBlockSizesForSimpleMatrixOperation(
  int32_t num_rows, int32_t num_cols, dim3 *dimGrid, dim3 *dimBlock);

inline int32_t n_blocks(int32_t size, int32_t block_size);

// Merely performs a static_cast
template<typename OutReal, typename InReal>
__global__
static void _copy_from_mat(OutReal* mat_out, const InReal* mat_in,
    RivaASRLibFeat_ContainerIndex_t cols, RivaASRLibFeat_ContainerIndex_t rows,
    RivaASRLibFeat_ContainerIndex_t stride_out, RivaASRLibFeat_ContainerIndex_t stride_in);

template<typename Real>
__global__
static void _compute_dct_matrix(Real* mat,
    RivaASRLibFeat_ContainerIndex_t cols, RivaASRLibFeat_ContainerIndex_t rows,
    RivaASRLibFeat_ContainerIndex_t stride);

template<typename Real>
CuMatrix<Real>::CuMatrix(){
  // Make sure we use device 0.
  riva::asrlib::feat::CuDevice::GetInstanceForThread();
}

template <typename Real>
CuMatrix<Real>::CuMatrix(
    RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
    RivaASRLibFeat_ContainerResizeOpt resize_type,
    RivaASRLibFeat_MatrixStrideOpt stride_type)
    : CuMatrix<Real>()
{
  Resize(rows, cols, resize_type, stride_type);
}

template <typename Real>
CuMatrix<Real>::CuMatrix(const CuMatrix<Real>& other)
{
  this->CopyFromMat(other);
}

template<typename Real>
CuMatrix<Real>::CuMatrix(CuMatrix<Real> &&other) noexcept
{
  Swap(other);
}

template<typename Real>
CuMatrix<Real>& CuMatrix<Real>::operator=(const CuMatrix<Real>& other)
{
  CuMatrix<Real> temp(other);
  Swap(temp);
  return *this;
}

template<typename Real>
CuMatrix<Real>& CuMatrix<Real>::operator=(CuMatrix<Real> &&other) noexcept
{
  Swap(other);
  return *this;
}

template<typename Real>
void CuMatrix<Real>::Swap(CuMatrix<Real> &other) noexcept
{
  std::swap(data_, other.data_);
  std::swap(num_cols_, other.num_cols_);
  std::swap(num_rows_, other.num_rows_);
  std::swap(stride_, other.stride_);
  std::swap(capacity_, other.capacity_);
}

// NOTE: Disable hetergeneous cpy ctor by default. Could change in future.
template <typename Real>
template <typename OtherReal>
CuMatrix<Real>::CuMatrix(const CuMatrix<OtherReal> &other)
{
  assert(0);
}

// hetergeneous cpy ctor for float = double
template <>
template <>
CuMatrix<float>::CuMatrix(const CuMatrix<double> &other)
{

  Resize(other.NumRows(), other.NumCols(),
    RivaASRLibFeat_ContainerResizeOpt::RivaASRLibFeat_kUndefined,
    RivaASRLibFeat_MatrixStrideOpt::RivaASRLibFeat_kDefaultStride);

  if(NumRows() == 0 || NumCols() == 0){
    assert(NumRows() == 0);
    assert(NumCols() == 0);
    return;
  }

  dim3 dimGrid, dimBlock;
  GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
    &dimGrid, &dimBlock);

  _copy_from_mat<<<dimGrid, dimBlock, 0, cudaStreamPerThread>>>(data_, other.Data(),
    num_cols_, num_rows_, stride_, other.Stride());

  CU_SAFE_CALL(cudaGetLastError());
  // NOTE: potential problem: any sync needed? Not seeing any in kaldi usage.
}


template <typename Real>
void
CuMatrix<Real>::Resize(
    RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
    RivaASRLibFeat_ContainerResizeOpt resize_type,
    RivaASRLibFeat_MatrixStrideOpt stride_type)
{
  // This code does not currently support the other resize_type options.
  assert(resize_type == RivaASRLibFeat_kSetZero || resize_type == RivaASRLibFeat_kUndefined);

  if (rows * cols == 0)
    assert(rows == 0 && cols == 0);

  RivaASRLibFeat_ContainerIndex_t new_stride;
  RivaASRLibFeat_ContainerIndex_t row_bytes = cols * sizeof(Real);
  if (stride_type == RivaASRLibFeat_kDefaultStride) {
    row_bytes = (row_bytes + 255) & ~((size_t)255);
    new_stride = row_bytes / sizeof(Real);
  } else {
    new_stride = cols;
  }

  if (rows * new_stride <= this->capacity_) {
    if (resize_type == RivaASRLibFeat_kSetZero)
      this->SetZero();

    // Update rows, cols and stride
    this->num_rows_ = rows;
    this->num_cols_ = cols;
    this->stride_ = new_stride;

    return;
  }

  if (rows * new_stride > this->capacity_ && this->capacity_ != 0)
    this->Destroy();

  if (rows == 0)
    return;

  if (stride_type == RivaASRLibFeat_kDefaultStride) {
    void* data;
    // Round up row bytes to multiple of 256
    row_bytes = (row_bytes + 255) & ~((size_t)255);
    // TODO: change to async after l4t gets cuda11 support.
    CU_SAFE_CALL(cudaMalloc(&data, row_bytes * rows));
    this->data_ = static_cast<Real*>(data);
    this->stride_ = row_bytes / sizeof(Real);
  } else {  // RivaASRLibFeat_kStrideEqualNumCols
    size_t bytes = rows * cols * sizeof(Real);
    bytes = (bytes + 255) & ~((size_t)255);
    void* data;
    // TODO: change to async after l4t gets cuda11 support.
    CU_SAFE_CALL(cudaMalloc(&data, bytes));
    this->data_ = static_cast<Real*>(data);
    this->stride_ = cols;
  }

  this->num_rows_ = rows;
  this->num_cols_ = cols;
  this->capacity_ = this->num_rows_ * this->stride_;

  if (resize_type == RivaASRLibFeat_kSetZero)
    this->SetZero();
}

template <typename Real>
void
CuMatrix<Real>::SetZero()
{
  CU_SAFE_CALL(cudaMemset2DAsync(
      data_, stride_ * sizeof(Real), 0, num_cols_ * sizeof(Real), num_rows_, cudaStreamPerThread));
}

template <typename Real>
void
CuMatrix<Real>::CopyFromMat(std::vector<std::vector<Real>>& mat, RivaASRLibFeat_ContainerIndex_t num_rows)
{
  assert(mat.size() >= num_rows);
  RivaASRLibFeat_ContainerIndex_t num_cols = mat[0].size();

  Resize(num_rows, num_cols,
    RivaASRLibFeat_ContainerResizeOpt::RivaASRLibFeat_kUndefined,
    RivaASRLibFeat_MatrixStrideOpt::RivaASRLibFeat_kDefaultStride);

  for (RivaASRLibFeat_ContainerIndex_t row = 0; row < num_rows; row++) {
    CU_SAFE_CALL(cudaMemcpyAsync(
        RowData(row), mat[row].data(), num_cols * sizeof(Real),
        cudaMemcpyHostToDevice, cudaStreamPerThread));
  }
}

template <typename Real>
void
CuMatrix<Real>::CopyFromMat(const CuMatrix<Real>& M)
{
  // NOTE: previously required user to resize if necessary before calling.
  Resize(M.NumRows(), M.NumCols(),
    RivaASRLibFeat_ContainerResizeOpt::RivaASRLibFeat_kUndefined,
    RivaASRLibFeat_MatrixStrideOpt::RivaASRLibFeat_kDefaultStride);

  assert(M.NumRows() == num_rows_ && M.NumCols() == num_cols_);

  if((NumCols() * NumRows()) == 0){
    assert(NumCols() == 0);
    assert(NumRows() == 0);
    return;
  }

  RivaASRLibFeat_ContainerIndex_t dst_pitch = stride_ * sizeof(Real);
  RivaASRLibFeat_ContainerIndex_t src_pitch = M.Stride() * sizeof(Real);
  RivaASRLibFeat_ContainerIndex_t width = M.NumCols() * sizeof(Real);
  CU_SAFE_CALL(cudaMemcpy2DAsync(
      data_, dst_pitch, M.data_, src_pitch, width, M.num_rows_, cudaMemcpyDeviceToDevice,
      cudaStreamPerThread));
  CU_SAFE_CALL(cudaGetLastError());
}

template <typename Real>
void
CuMatrix<Real>::CopyFromMat(const CuMatrix<Real>& M, int num_rows)
{

  // NOTE: previously required user to resize if necessary before calling.
  Resize(num_rows, M.NumCols(),
    RivaASRLibFeat_ContainerResizeOpt::RivaASRLibFeat_kUndefined,
    RivaASRLibFeat_MatrixStrideOpt::RivaASRLibFeat_kDefaultStride);

  assert(M.NumCols() == num_cols_);

  if((NumCols() * NumRows()) == 0){
    assert(NumCols() == 0);
    assert(NumRows() == 0);
    return;
  }

  RivaASRLibFeat_ContainerIndex_t dst_pitch = stride_ * sizeof(Real);
  RivaASRLibFeat_ContainerIndex_t src_pitch = M.Stride() * sizeof(Real);
  RivaASRLibFeat_ContainerIndex_t width = M.NumCols() * sizeof(Real);
  CU_SAFE_CALL(cudaMemcpy2DAsync(
      data_, dst_pitch, M.data_, src_pitch, width, num_rows, cudaMemcpyDeviceToDevice,
      cudaStreamPerThread));
  CU_SAFE_CALL(cudaGetLastError());
}

template<typename Real>
void
CuMatrix<Real>::CopyFromMat(
    const Real *other_data_cpu,
    RivaASRLibFeat_ContainerIndex_t other_rows, RivaASRLibFeat_ContainerIndex_t other_cols,
    RivaASRLibFeat_ContainerIndex_t other_stride)
{
  Resize(other_rows, other_cols,
    RivaASRLibFeat_ContainerResizeOpt::RivaASRLibFeat_kUndefined,
    RivaASRLibFeat_MatrixStrideOpt::RivaASRLibFeat_kDefaultStride);

  if((NumCols() * NumRows()) == 0){
    assert(NumCols() == 0);
    assert(NumRows() == 0);
    return;
  }

  auto dst_pitch = stride_ * sizeof(Real);
  auto src_pitch = other_stride * sizeof(Real);
  auto width = other_cols * sizeof(Real);
  CU_SAFE_CALL(cudaMemcpy2DAsync(
      data_, dst_pitch,
      other_data_cpu, src_pitch,
      width, other_rows, cudaMemcpyHostToDevice,
      cudaStreamPerThread));
  CU_SAFE_CALL(cudaGetLastError());
  // TODO: potential problem: any sync needed?
}

template <typename Real>
void
CuMatrix<Real>::ComputeDctMatrix(){

  RivaASRLibFeat_ContainerIndex_t K = NumRows();
  RivaASRLibFeat_ContainerIndex_t N = NumCols();

  std::vector<std::vector<Real>> vv(
    NumRows(),
    std::vector<Real>(NumCols(), 0)
  );

  assert(K > 0);
  assert(N > 0);

  Real normalizer = std::sqrt(1.0 / static_cast<Real>(N));  // normalizer for
  // X_0.
  for (RivaASRLibFeat_ContainerIndex_t j = 0; j < N; j++){
    vv[0][j] = normalizer;
  }

  normalizer = std::sqrt(2.0 / static_cast<Real>(N));  // normalizer for other
   // elements.
  for (RivaASRLibFeat_ContainerIndex_t k = 1; k < K; k++)
    for (RivaASRLibFeat_ContainerIndex_t n = 0; n < N; n++)
      vv[k][n] = normalizer
          * std::cos( static_cast<double>(M_PI)/N * (n + 0.5) * k );

  CopyFromMat(vv, NumRows());

  /*
  // kernel version:
  dim3 dimGrid, dimBlock;
  GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
    &dimGrid, &dimBlock);

  _compute_dct_matrix<<<dimGrid, dimBlock>>>(
    data_, num_cols_, num_rows_, stride_);

  CU_SAFE_CALL(cudaGetLastError());
  */
  // NOTE: potential problem: any sync needed? Not seeing any in kaldi usage.
}


template <typename Real>
void
CuMatrix<Real>::AddMatMat(
    Real alpha, const CuMatrix<Real>& A, RivaASRLibFeat_MatrixTransposeOpt transA, const CuMatrix<Real>& B,
    RivaASRLibFeat_MatrixTransposeOpt transB, Real beta, cublasHandle_t& cublas_handle)
{
  // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
  // keep trans..., just swap A&B matrices: A->B B->A
  RivaASRLibFeat_ContainerIndex_t m = ((transB == RivaASRLibFeat_kTrans) ? B.NumRows() : B.NumCols());
  RivaASRLibFeat_ContainerIndex_t n = ((transA == RivaASRLibFeat_kTrans) ? A.NumCols() : A.NumRows());
  RivaASRLibFeat_ContainerIndex_t k = ((transB == RivaASRLibFeat_kTrans) ? B.NumCols() : B.NumRows());

  assert(m == NumCols());
  assert(n == NumRows());
  assert(k == ((transA == RivaASRLibFeat_kTrans) ? A.NumRows() : A.NumCols()));

  if (m == 0)
    return;

  CUBLAS_SAFE_CALL(cublas_gemm(
      cublas_handle, (transB == RivaASRLibFeat_kTrans ? CUBLAS_OP_T : CUBLAS_OP_N),
      (transA == RivaASRLibFeat_kTrans ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, alpha, B.data_, B.Stride(), A.data_,
      A.Stride(), beta, data_, Stride()));
}


template <>
void
CuMatrix<uint8_t>::AddMatMat(
    uint8_t alpha,
    const CuMatrix<uint8_t>& A, RivaASRLibFeat_MatrixTransposeOpt transA,
    const CuMatrix<uint8_t>& B, RivaASRLibFeat_MatrixTransposeOpt transB,
    uint8_t beta,
    cublasHandle_t& cublas_handle)
{
  assert(0);
}

// NOTE: Added this one specialized function instead of implementing entire sub matrix view functionality.
template <typename Real>
void
CuMatrix<Real>::AddMatMatIgnoreDestNTrailingRows(
    Real alpha,
    const CuMatrix<Real>& A, RivaASRLibFeat_MatrixTransposeOpt transA,
    const CuMatrix<Real>& B, RivaASRLibFeat_MatrixTransposeOpt transB,
    Real beta, cublasHandle_t& cublas_handle,
    RivaASRLibFeat_ContainerIndex_t num_trailing_rows_ignored)
{
  // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
  // keep trans..., just swap A&B matrices: A->B B->A
  RivaASRLibFeat_ContainerIndex_t m = ((transB == RivaASRLibFeat_kTrans) ? B.NumRows() : B.NumCols());
  RivaASRLibFeat_ContainerIndex_t n = ((transA == RivaASRLibFeat_kTrans) ? A.NumCols() : A.NumRows());
  RivaASRLibFeat_ContainerIndex_t k = ((transB == RivaASRLibFeat_kTrans) ? B.NumCols() : B.NumRows());

  auto effective_num_rows = NumRows() - num_trailing_rows_ignored;

  assert(m == NumCols());
  assert(n == effective_num_rows);
  assert(k == ((transA == RivaASRLibFeat_kTrans) ? A.NumRows() : A.NumCols()));

  if (m == 0)
    return;

  CUBLAS_SAFE_CALL(cublas_gemm(
      cublas_handle, (transB == RivaASRLibFeat_kTrans ? CUBLAS_OP_T : CUBLAS_OP_N),
      (transA == RivaASRLibFeat_kTrans ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, alpha, B.data_, B.Stride(), A.data_,
      A.Stride(), beta, data_, Stride()));
}

template <>
void
CuMatrix<uint8_t>::AddMatMatIgnoreDestNTrailingRows(
    uint8_t alpha,
    const CuMatrix<uint8_t>& A, RivaASRLibFeat_MatrixTransposeOpt transA,
    const CuMatrix<uint8_t>& B, RivaASRLibFeat_MatrixTransposeOpt transB,
    uint8_t beta, cublasHandle_t& cublas_handle,
    RivaASRLibFeat_ContainerIndex_t num_trailing_rows_ignored)
{
  assert(0);
}


template <typename Real>
void
CuMatrix<Real>::Print(std::string& filename)
{
  std::ofstream file;

  std::vector<Real> host(num_rows_ * stride_);
  CU_SAFE_CALL(
      cudaMemcpy(&host[0], data_, num_rows_ * stride_ * sizeof(Real), cudaMemcpyDeviceToHost));

  file.open(filename);
  for (RivaASRLibFeat_ContainerIndex_t row = 0; row < num_rows_; row++) {
    for (RivaASRLibFeat_ContainerIndex_t col = 0; col < num_cols_; col++) {
      file << host[row * stride_ + col];
      if (col != num_cols_ - 1) {
        file << ", ";
      }
    }
    file << std::endl;
  }
  file.close();
}

template <typename Real>
void
CuMatrix<Real>::Destroy()
{
  if (this->data_ != NULL) {
    // TODO: change to async after l4t gets cuda11 support.
    CU_SAFE_CALL(cudaFree(this->data_));
  }

  this->data_ = NULL;
  this->num_rows_ = 0;
  this->num_cols_ = 0;
  this->stride_ = 0;
  this->capacity_ = 0;
}

void GetBlockSizesForSimpleMatrixOperation(int32_t num_rows,
                                           int32_t num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock) {
  assert(num_rows > 0 && num_cols > 0);
  int32_t col_blocksize = 64, row_blocksize = 4;
  while (col_blocksize > 1 &&
         (num_cols + (num_cols / 2) <= col_blocksize ||
          num_rows > 65535 * row_blocksize)) {
    col_blocksize /= 2;
    row_blocksize *= 2;
  }

  dimBlock->x = col_blocksize;
  dimBlock->y = row_blocksize;
  dimBlock->z = 1;
  dimGrid->x = n_blocks(num_cols, col_blocksize);
  dimGrid->y = n_blocks(num_rows, row_blocksize);
  assert(dimGrid->y <= 65535 &&
               "Matrix has too many rows to process");
  dimGrid->z = 1;
}

inline int32_t n_blocks(int32_t size, int32_t block_size) {
    return size / block_size + ((size % block_size == 0)? 0 : 1);
}

template<typename OutReal, typename InReal>
__global__
void _copy_from_mat(OutReal* mat_out, const InReal* mat_in,
    RivaASRLibFeat_ContainerIndex_t cols, RivaASRLibFeat_ContainerIndex_t rows,
    RivaASRLibFeat_ContainerIndex_t stride_out, RivaASRLibFeat_ContainerIndex_t stride_in){

  int32_cuda col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda index_out = col_idx + row_idx * stride_out;
  int32_cuda index_in = col_idx + row_idx * stride_in;
  if (col_idx < cols && row_idx < rows)
    mat_out[index_out] = static_cast<OutReal>(mat_in[index_in]);
}

template<typename Real>
__global__
void _compute_dct_matrix(Real* mat,
    RivaASRLibFeat_ContainerIndex_t cols, RivaASRLibFeat_ContainerIndex_t rows,
    RivaASRLibFeat_ContainerIndex_t stride){

  int32_cuda col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_cuda row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int32_cuda data_idx = col_idx + row_idx * stride;

  if(col_idx >= cols || row_idx >= rows){
    return;
  }

  if(row_idx == 0){
    Real normalizer = sqrt(1.0 / static_cast<Real>(cols));  // normalizer for X_0
    mat[data_idx] = normalizer;
  }
  else{
    Real normalizer = sqrt(2.0 / static_cast<Real>(cols));  // normalizer for other
    mat[data_idx] = normalizer * cos(static_cast<double>(M_PI) / cols * (col_idx + 0.5) * row_idx);
  }
}

// Explicit instantiation
template class CuMatrix<float>;
template class CuMatrix<double>;
template class CuMatrix<uint8_t>;

} // namespace feat
} // namespace asrlib
} // namespace riva
