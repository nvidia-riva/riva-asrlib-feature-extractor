// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely.
// SPDX-FileCopyrightText: Copyright (c) 2013       Johns Hopkins University (author: Daniel Povey), Hainan Xu, Xiaohui Zhang
// SPDX-FileCopyrightText: Copyright (c) 2013-2015  Guoguo Chen
// SPDX-FileCopyrightText: Copyright (c) 2017       Shiyin Kang
// SPDX-FileCopyrightText: Copyright (c) 2019       Yiwen Shao
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_CU_MATRIX_H_
#define RIVA_ASRLIB_FEAT_CU_MATRIX_H_

#include <assert.h>

#include <vector>
#include <string>

#include "cu-feat-extr/c-cpp-bridge/cu-container-helper-types.h"

#include <cublas_v2.h>

namespace riva {
namespace asrlib {
namespace feat {


template <typename Real>
class CuMatrix {
 public:
  RivaASRLibFeat_ContainerIndex_t NumRows() const { return num_rows_; }
  RivaASRLibFeat_ContainerIndex_t NumCols() const { return num_cols_; }
  RivaASRLibFeat_ContainerIndex_t Stride() const { return stride_; }

  inline const Real* Data() const { return data_; }
  inline Real* Data() { return data_; }
  inline const Real* RowData(RivaASRLibFeat_ContainerIndex_t r) const { return data_ + r * stride_; }
  inline Real* RowData(RivaASRLibFeat_ContainerIndex_t r) { return data_ + r * stride_; }

  CuMatrix();

  /// Constructor with memory initialisation
  CuMatrix(
      RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
      RivaASRLibFeat_ContainerResizeOpt resize_type = RivaASRLibFeat_kSetZero,
      RivaASRLibFeat_MatrixStrideOpt stride_type = RivaASRLibFeat_kDefaultStride);

  CuMatrix(const CuMatrix<Real> &other);
  CuMatrix(CuMatrix<Real> &&other) noexcept;

  CuMatrix<Real>& operator=(const CuMatrix<Real>& other);
  CuMatrix<Real>& operator=(CuMatrix<Real> &&other) noexcept;

  void Swap(CuMatrix<Real> &other) noexcept;

  template <typename OtherReal>
  CuMatrix(const CuMatrix<OtherReal> &other);

  void Resize(
      RivaASRLibFeat_ContainerIndex_t rows, RivaASRLibFeat_ContainerIndex_t cols,
      RivaASRLibFeat_ContainerResizeOpt resize_type = RivaASRLibFeat_kSetZero,
      RivaASRLibFeat_MatrixStrideOpt stride_type = RivaASRLibFeat_kDefaultStride);

  void SetZero();

  void AddMatMat(
      Real alpha,
      const CuMatrix<Real>& A, RivaASRLibFeat_MatrixTransposeOpt transA,
      const CuMatrix<Real>& B, RivaASRLibFeat_MatrixTransposeOpt transB,
      Real beta, cublasHandle_t& cublas_handle);

  void AddMatMatIgnoreDestNTrailingRows(
      Real alpha,
      const CuMatrix<Real>& A, RivaASRLibFeat_MatrixTransposeOpt transA,
      const CuMatrix<Real>& B, RivaASRLibFeat_MatrixTransposeOpt transB,
      Real beta, cublasHandle_t& cublas_handle,
      RivaASRLibFeat_ContainerIndex_t num_trailing_rows_ignored);

  void CopyFromMat(std::vector<std::vector<Real>>& mat, RivaASRLibFeat_ContainerIndex_t num_rows);

  void CopyFromMat(const CuMatrix<Real>& src);

  void CopyFromMat(const CuMatrix<Real>& src, int num_rows);

  // Overload exists to avoid code duplication in C API
  // other_data_cpu must be row major AND LIVE IN CPU MEMORY.
  void CopyFromMat(const Real *other_data_cpu,
    RivaASRLibFeat_ContainerIndex_t other_rows,
    RivaASRLibFeat_ContainerIndex_t other_cols,
    RivaASRLibFeat_ContainerIndex_t other_stride);

  void ComputeDctMatrix();

  void Print(std::string& filename);

  void Destroy();

  ~CuMatrix() { Destroy(); }

 private:
  Real* data_ = NULL;  ///< GPU data pointer (or regular matrix data pointer,
  RivaASRLibFeat_ContainerIndex_t num_cols_ = 0;
  RivaASRLibFeat_ContainerIndex_t num_rows_ = 0;
  RivaASRLibFeat_ContainerIndex_t stride_ = 0;
  RivaASRLibFeat_ContainerIndex_t capacity_ = 0;
};

} // namespace feat
} // namespace asrlib
} // namespace riva


#endif
