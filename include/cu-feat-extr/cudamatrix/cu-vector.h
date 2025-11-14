// SPDX-FileCopyrightText: Copyright (c) 2009-2012  Karel Vesely, Johns Hopkins University (author: Daniel Povey), Lucas Ondel
// SPDX-FileCopyrightText: Copyright (c) 2013       Xiaohui Zhang
// SPDX-FileCopyrightText: Copyright (c) 2015       Guoguo Chen
// SPDX-FileCopyrightText: Copyright (c) 2017       Daniel Galvez
// SPDX-FileCopyrightText: Copyright (c) 2019       Yiwen Shao
// SPDX-FileCopyrightText: Copyright (c) 2025       NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef RIVA_ASRLIB_FEAT_CU_VECTOR_H_
#define RIVA_ASRLIB_FEAT_CU_VECTOR_H_

#include "cu-feat-extr/c-cpp-bridge/cu-container-helper-types.h"
#include <vector>
#include <string>

namespace riva {
namespace asrlib {
namespace feat {

template <typename Real>
class CuVector {
 public:

  RivaASRLibFeat_ContainerIndex_t Dim() const { return dim_; }

  inline Real* Data() { return data_; }
  inline const Real* Data() const { return data_; }

  CuVector() : data_(NULL), dim_(0), capacity_(0) {}

  /// Constructor with memory initialisation
  CuVector(RivaASRLibFeat_ContainerIndex_t size, RivaASRLibFeat_ContainerResizeOpt resize_type = RivaASRLibFeat_kSetZero);

  CuVector(const CuVector<Real> &other) = delete;
  CuVector(CuVector<Real> &&other) = default;

  CuVector& operator=(const CuVector<Real> &other) = delete;
  CuVector& operator=(CuVector<Real> &&other) = default;

  void SetZero();

  void Print(std::string& filename);

  // Function checks if entries from 0 to n-1 are different between
  // host new_vector and old_vector. If so, updates old_vector and cu_vector
  void CheckIfChangedAndCopy(std::vector<Real>& new_vector, std::vector<Real>& old_vector, int n);

  void Resize(RivaASRLibFeat_ContainerIndex_t dim, RivaASRLibFeat_ContainerResizeOpt t = RivaASRLibFeat_kSetZero);

  void CopyFromVec(const std::vector<Real>& vec);

  void CopyFromVec(const std::vector<Real>& vec, int size);

  void CopyFromVec(const std::vector<Real>& vec, int origin, int size);

  void CopyFromVec(const CuVector<Real> &vec);

  void CopyToVec(std::vector<Real>& vec);

  ~CuVector() { Destroy(); }

 private:
  void Destroy();

  Real* data_;
  RivaASRLibFeat_ContainerIndex_t dim_;
  RivaASRLibFeat_ContainerIndex_t capacity_;
};

} // namespace feat
} // namespace asrlib
} // namespace riva

#endif
