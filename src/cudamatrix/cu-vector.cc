// SPDX-FileCopyrightText: Copyright (c) 2012-2013  Karel Vesely
// SPDX-FileCopyrightText: Copyright (c) 2012-2014  Johns Hopkins University (author: Daniel Povey)
// SPDX-FileCopyrightText: Copyright (c)      2017  Daniel Galvez
// SPDX-FileCopyrightText: Copyright (c) 2016-2018  Shiyin Kang
// SPDX-FileCopyrightText: Copyright (c)      2019  Yiwen Shao
// SPDX-FileCopyrightText: Copyright (c)      2025  NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cu-feat-extr/cudamatrix/cu-vector.h"

#include <fstream>
#include <iostream>
#include <cassert>

#include <cuda_runtime_api.h>
#include "cu-feat-extr/cudamatrix/cu-common.h"

namespace riva {
namespace asrlib {
namespace feat {

/// Constructor with memory initialisation
template <typename Real>
CuVector<Real>::CuVector(RivaASRLibFeat_ContainerIndex_t size, RivaASRLibFeat_ContainerResizeOpt resize_type)
    : data_(NULL), dim_(0), capacity_(0)
{
  Resize(size, resize_type);
}


template <typename Real>
void
CuVector<Real>::Print(std::string& filename)
{
  std::ofstream file;

  std::vector<Real> host(dim_);
  CU_SAFE_CALL(cudaMemcpy(&host[0], data_, dim_ * sizeof(Real), cudaMemcpyDeviceToHost));

  file.open(filename);
  for (RivaASRLibFeat_ContainerIndex_t col = 0; col < dim_; col++) {
    file << host[col];
    if (col != dim_ - 1) {
      file << std::endl;
    }
  }
  file.close();
}

template <typename Real>
void
CuVector<Real>::SetZero()
{
  if (dim_ == 0 || data_ == NULL)
    return;

  assert(dim_ >= 0);
  assert(data_ != NULL);
  CU_SAFE_CALL(cudaMemsetAsync(data_, 0, dim_ * sizeof(Real), cudaStreamPerThread));
}

template <typename Real>
void
CuVector<Real>::CopyFromVec(const std::vector<Real>& vec)
{
  if (vec.size() == 0) {
    return;
  }

  this->Resize(vec.size(), RivaASRLibFeat_kUndefined);

  CU_SAFE_CALL(cudaMemcpyAsync(
      this->data_, &vec[0], vec.size() * sizeof(Real), cudaMemcpyHostToDevice,
      cudaStreamPerThread));
}

template <typename Real>
void
CuVector<Real>::CopyToVec(std::vector<Real>& vec)
{
  vec.resize(dim_);

  CU_SAFE_CALL(cudaMemcpyAsync(
      &vec[0], this->data_, vec.size() * sizeof(Real), cudaMemcpyDeviceToHost,
      cudaStreamPerThread));
}


template <typename Real>
void
CuVector<Real>::CopyFromVec(const std::vector<Real>& vec, int size)
{
  if (size == 0) {
    return;
  }

  this->Resize(size, RivaASRLibFeat_kUndefined);

  CU_SAFE_CALL(cudaMemcpyAsync(
      this->data_, &vec[0], size * sizeof(Real), cudaMemcpyHostToDevice, cudaStreamPerThread));
}

template <typename Real>
void
CuVector<Real>::CopyFromVec(const std::vector<Real>& vec, int origin, int size)
{
  if (size == 0) {
    return;
  }

  this->Resize(size, RivaASRLibFeat_kUndefined);

  CU_SAFE_CALL(cudaMemcpyAsync(
      this->data_, &vec[origin], size * sizeof(Real), cudaMemcpyHostToDevice, cudaStreamPerThread));
}

template <typename Real>
void
CuVector<Real>::CopyFromVec(const CuVector<Real> &vec)
{
  if (vec.Dim() == 0) {
    return;
  }

  this->Resize(vec.Dim(), RivaASRLibFeat_kUndefined);

  CU_SAFE_CALL(cudaMemcpyAsync(
      this->data_, vec.Data(), vec.Dim() * sizeof(Real), cudaMemcpyHostToDevice,
      cudaStreamPerThread));
}

template <typename Real>
void
CuVector<Real>::CheckIfChangedAndCopy(
    std::vector<Real>& new_vector, std::vector<Real>& old_vector, int n)
{
  bool changed = false;
  for (int i = 0; i < n; i++) {
    if (new_vector[i] != old_vector[i]) {
      old_vector[i] = new_vector[i];
      changed = true;
    }
  }
  if (changed) {
    this->CopyFromVec(old_vector);
  }
}


template <typename Real>
void
CuVector<Real>::Resize(RivaASRLibFeat_ContainerIndex_t dim, RivaASRLibFeat_ContainerResizeOpt t)
{
  assert(t == RivaASRLibFeat_kSetZero || t == RivaASRLibFeat_kUndefined);

  if (dim <= this->capacity_) {
    if (t == RivaASRLibFeat_kSetZero)
      this->SetZero();
    this->dim_ = dim;
    return;
  }

  if (dim > this->capacity_ && this->capacity_ != 0)
    this->Destroy();

  if (dim == 0)
    return;

  void* data;
  size_t bytes = dim * sizeof(Real);
  // Round up to multiple of 256
  bytes = (bytes + 255) & ~((size_t)255);
  // TODO: change to async after l4t gets cuda11 support.
  CU_SAFE_CALL(cudaMalloc(&data, bytes));
  this->data_ = static_cast<Real*>(data);
  this->dim_ = dim;
  this->capacity_ = dim;
  if (t == RivaASRLibFeat_kSetZero)
    this->SetZero();
}

template <typename Real>
void
CuVector<Real>::Destroy()
{
  if (this->data_ != NULL)
    // TODO: change to async after l4t gets cuda11 support.
    CU_SAFE_CALL(cudaFree(this->data_));

  this->data_ = NULL;
  this->dim_ = 0;
  this->capacity_ = 0;
}

// Explicit instantiation
template class CuVector<float>;
template class CuVector<float*>;
template class CuVector<const float*>;
template class CuVector<int32_t>;
template class CuVector<uint32_t>;
template class CuVector<uint8_t>;

} // namespace feat
} // namespace asrlib
} // namespace riva