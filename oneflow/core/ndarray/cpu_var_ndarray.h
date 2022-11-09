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
#ifndef ONEFLOW_CORE_NDARRAY_CPU_VAR_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_CPU_VAR_NDARRAY_H_

#include "oneflow/core/ndarray/cpu_ndarray.h"
#include "oneflow/core/ndarray/cpu_ndarray_copy.h"

namespace oneflow {

class Slice;
template<typename XT>
class CpuSliceVarNdarray;

template<typename T, int NDIMS>
class CpuVarNdarray : public CpuNdarray<T, NDIMS> {
 public:
  CpuVarNdarray(const CpuVarNdarray&) = default;
  CpuVarNdarray(const Shape& shape, T* ptr)
      : CpuNdarray<T, NDIMS>(shape), ptr_(ptr), len_(shape.elem_cnt()) {
    CHECK_GT(len_, 0);
  }
  CpuVarNdarray(const ShapeView& shape_view, T* ptr)
      : CpuNdarray<T, NDIMS>(XpuShape(shape_view)), ptr_(ptr), len_(shape_view.elem_cnt()) {
    CHECK_GT(len_, 0);
  }
  virtual ~CpuVarNdarray() = default;

  CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>> operator()(Slice&& slice0) {
    static_assert(NDIMS == 1, "NDIMS error");
    return CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>>(std::move(*this), {slice0});
  }
  CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>> operator()(Slice&& slice0, Slice&& slice1) {
    static_assert(NDIMS == 2, "NDIMS error");
    return CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>>(std::move(*this), {slice0, slice1});
  }
  CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>> operator()(Slice&& slice0, Slice&& slice1,
                                                         Slice&& slice2) {
    static_assert(NDIMS == 3, "NDIMS error");
    return CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>>(std::move(*this), {slice0, slice1, slice2});
  }
  CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>> operator()(Slice&& slice0, Slice&& slice1,
                                                         Slice&& slice2, Slice&& slice3) {
    static_assert(NDIMS == 4, "NDIMS error");
    return CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>>(std::move(*this),
                                                       {slice0, slice1, slice2, slice3});
  }
  CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>> operator()(Slice&& slice0, Slice&& slice1,
                                                         Slice&& slice2, Slice&& slice3,
                                                         Slice&& slice4) {
    static_assert(NDIMS == 5, "NDIMS error");
    return CpuSliceVarNdarray<CpuVarNdarray<T, NDIMS>>(std::move(*this),
                                                       {slice0, slice1, slice2, slice3, slice4});
  }

  template<typename XT>
  void CopyFrom(const XT& ndarray) {
    CpuNdarrayCopy(this, ndarray);
  }

  ALWAYS_INLINE void GetMutPtrAndContiguousSize(int64_t offset, T** ptr, size_t* size) const {
    *ptr = ptr_ + offset;
    *size = len_ - offset;
  }

 protected:
  ALWAYS_INLINE T* ptr() const { return ptr_; }
  ALWAYS_INLINE size_t len() const { return len_; }

 private:
  T* const ptr_;
  size_t len_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_CPU_VAR_NDARRAY_H_
