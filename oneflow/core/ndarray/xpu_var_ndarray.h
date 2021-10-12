/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_

#include "oneflow/core/ndarray/xpu_shape.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/ndarray/xpu_ndarray_base.h"

namespace oneflow {

template<typename T>
class XpuVarNdarray final : public XpuNdarrayBase<XpuVarNdarray<T>, T> {
 public:
  XpuVarNdarray(const Blob* blob, int ndims_left_extend_to)
      : shape_(blob->shape(), ndims_left_extend_to),
        ptr_(blob->dptr<typename std::remove_const<T>::type>()) {}
  XpuVarNdarray(Blob* blob, int ndims_left_extend_to)
      : shape_(blob->shape(), ndims_left_extend_to), ptr_(blob->mut_dptr<T>()) {}
  XpuVarNdarray(const Shape& shape, T* ptr) : shape_(shape), ptr_(ptr) {}
  XpuVarNdarray(const ShapeView& shape, T* ptr) : shape_(shape), ptr_(ptr) {}
  XpuVarNdarray(const ShapeView& shape, T* ptr, int ndims_left_extend_to)
      : shape_(shape, ndims_left_extend_to), ptr_(ptr) {}
  ~XpuVarNdarray() = default;
  ALWAYS_INLINE XpuVarNdarray(const XpuVarNdarray&) = default;
  OF_DEVICE_FUNC ALWAYS_INLINE XpuVarNdarray(const XpuShape& shape, T* ptr)
      : shape_(shape), ptr_(ptr) {}

  const XpuShape& host_shape() const { return shape_; }
  T* host_ptr() const { return ptr_; }

  OF_DEVICE_FUNC const XpuShape& shape() const { return shape_; }
  OF_DEVICE_FUNC T* ptr() const { return ptr_; }

  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    return ptr_[offset];
  }
  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t coord[NDIMS]) const {
    return ptr_[shape().template Coordinate2Offset<NDIMS>(coord)];
  }

  template<int NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t offset) const {
    return ptr_ + offset;
  }

  template<int NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t coord[NDIMS]) const {
    return ptr_ + shape().template Coordinate2Offset<NDIMS>(coord);
  }

  template<int NDIMS, typename X>
  OF_DEVICE_FUNC void Assign(const X& x) const {
    size_t n = shape_.ElemNum();
    XPU_1D_KERNEL_LOOP_BEGIN(i, n);
    ptr_[i] = x.template Get<NDIMS>(i);
    XPU_1D_KERNEL_LOOP_END();
  }

  template<template<typename> class binary_func, int NDIMS, typename X>
  OF_DEVICE_FUNC void BinaryAssign(const X& x) const {
    size_t n = shape_.ElemNum();
    XPU_1D_KERNEL_LOOP_BEGIN(i, n);
    T* ptr_i = ptr_ + i;
    *ptr_i = binary_func<T>::Invoke(*ptr_i, x.template Get<NDIMS>(i));
    XPU_1D_KERNEL_LOOP_END();
  }

 private:
  XpuShape shape_;
  T* ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_
