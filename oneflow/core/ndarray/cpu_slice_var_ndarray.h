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
#ifndef ONEFLOW_CORE_NDARRAY_CPU_SLICE_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_CPU_SLICE_NDARRAY_H_

#include "oneflow/core/ndarray/slice.h"
#include "oneflow/core/ndarray/cpu_ndarray.h"
#include "oneflow/core/ndarray/cpu_ndarray_copy.h"

namespace oneflow {

template<typename XT>
class CpuSliceVarNdarray : public CpuNdarray<typename XT::dtype, XT::ndims> {
 public:
  CpuSliceVarNdarray(XT&& x, std::array<Slice, XT::ndims>&& slices)
      : CpuNdarray<typename XT::dtype, XT::ndims>(
          BoundedSlices2Shape(BoundSlices(x, std::move(slices)))),
        x_(x),
        slices_(std::move(slices)) {
    SetContiguousLength(slices);
  }
  virtual ~CpuSliceVarNdarray() = default;

  CpuSliceVarNdarray<CpuSliceVarNdarray<XT>> operator()(Slice&& slice0) {
    static_assert(XT::ndims == 1, "NDIMS error");
    return CpuSliceVarNdarray<CpuSliceVarNdarray<XT>>(std::move(*this), {slice0});
  }
  CpuSliceVarNdarray<CpuSliceVarNdarray<XT>> operator()(Slice&& slice0, Slice&& slice1) {
    static_assert(XT::ndims == 2, "NDIMS error");
    return CpuSliceVarNdarray<CpuSliceVarNdarray<XT>>(std::move(*this), {slice0, slice1});
  }
  CpuSliceVarNdarray<CpuSliceVarNdarray<XT>> operator()(Slice&& slice0, Slice&& slice1,
                                                        Slice&& slice2) {
    static_assert(XT::ndims == 3, "NDIMS error");
    return CpuSliceVarNdarray<CpuSliceVarNdarray<XT>>(std::move(*this), {slice0, slice1, slice2});
  }
  CpuSliceVarNdarray<CpuSliceVarNdarray<XT>> operator()(Slice&& slice0, Slice&& slice1,
                                                        Slice&& slice2, Slice&& slice3) {
    static_assert(XT::ndims == 4, "NDIMS error");
    return CpuSliceVarNdarray<CpuSliceVarNdarray<XT>>(std::move(*this),
                                                      {slice0, slice1, slice2, slice3});
  }
  CpuSliceVarNdarray<CpuSliceVarNdarray<XT>> operator()(Slice&& slice0, Slice&& slice1,
                                                        Slice&& slice2, Slice&& slice3,
                                                        Slice&& slice4) {
    static_assert(XT::ndims == 5, "NDIMS error");
    return CpuSliceVarNdarray<CpuSliceVarNdarray<XT>>(std::move(*this),
                                                      {slice0, slice1, slice2, slice3, slice4});
  }

  template<typename AT>
  void CopyFrom(const AT& ndarray) {
    CpuNdarrayCopy(this, ndarray);
  }

  ALWAYS_INLINE void GetMutPtrAndContiguousSize(int64_t offset, typename XT::dtype** ptr,
                                                size_t* size) const {
    int64_t dim[XT::ndims] = {0};
    this->xpu_shape().template Offset2Coordinate<XT::ndims>(offset, dim);
    for (int i = 0; i < XT::ndims; ++i) { dim[i] = this->slice(i).Get(dim[i]); }
    size_t x_offset = this->x().xpu_shape().template Coordinate2Offset<XT::ndims>(dim);
    this->GetMutPtrAndMinContiguousSize(offset, x_offset, ptr, size);
  }

 protected:
  ALWAYS_INLINE const XT& x() const { return x_; }
  ALWAYS_INLINE const Slice& slice(int32_t dim) const { return slices_[dim]; }
  ALWAYS_INLINE void GetMutPtrAndMinContiguousSize(int64_t offset, int64_t x_offset,
                                                   typename XT::dtype** ptr, size_t* size) const {
    size_t x_contiguous_size;
    this->x().GetMutPtrAndContiguousSize(x_offset, ptr, &x_contiguous_size);
    size_t slice_contiguous_size = (contiguous_len_ - offset % contiguous_len_);
    *size = std::min(x_contiguous_size, slice_contiguous_size);
  }

 private:
  static std::array<Slice, XT::ndims>&& BoundSlices(const XT& x,
                                                    std::array<Slice, XT::ndims>&& slices) {
    FOR_RANGE(int32_t, i, 0, XT::ndims) { slices[i].Bound(x.xpu_shape().At(i)); }
    return std::move(slices);
  }
  static Shape BoundedSlices2Shape(const std::array<Slice, XT::ndims>& bounded_slices) {
    DimVector dim_vec;
    for (const Slice& slice : bounded_slices) {
      CHECK_GT(slice.Size(), 0);
      dim_vec.push_back(slice.Size());
    }
    return Shape(dim_vec);
  }
  void SetContiguousLength(const std::array<Slice, XT::ndims>& bounded_slices) {
    contiguous_len_ = 1;
    for (int i = XT::ndims - 1; i >= 0; --i) {
      if (bounded_slices[i].IsContiguous()) { contiguous_len_ *= bounded_slices[i].Size(); }
      if (!(bounded_slices[i].IsContiguous() && bounded_slices[i].IsCoveringAll())) { break; }
    }
  }
  const XT& x_;
  std::array<Slice, XT::ndims> slices_;
  size_t contiguous_len_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_CPU_SLICE_NDARRAY_H_
