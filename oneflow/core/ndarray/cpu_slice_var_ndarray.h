#ifndef ONEFLOW_CORE_NDARRAY_SLICE_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_SLICE_NDARRAY_H_

#include "oneflow/core/ndarray/slice.h"
#include "oneflow/core/ndarray/cpu_ndarray.h"
#include "oneflow/core/ndarray/ndarray_copy.h"

namespace oneflow {

template<typename XT>
class CpuSliceVarNdArray : public CpuNdArray<typename XT::dtype, XT::ndims> {
 public:
  using dtype = typename XT::dtype;
  static const int ndims = XT::ndims;
  CpuSliceVarNdArray(XT&& x, std::array<Slice, ndims>&& slices)
      : CpuNdArray<typename XT::dtype, XT::ndims>(
            BoundedSlices2Shape(BoundSlices(x, std::move(slices)))),
        x_(x),
        slices_(std::move(slices)) {
    SetContiguousLength(slices);
  }
  virtual ~CpuSliceVarNdArray() = default;

  CpuSliceVarNdArray<CpuSliceVarNdArray<XT>> operator()(Slice&& slice0) {
    static_assert(ndims == 1, "NDIMS error");
    return CpuSliceVarNdArray<CpuSliceVarNdArray<XT>>(std::move(*this), {slice0});
  }
  CpuSliceVarNdArray<CpuSliceVarNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1) {
    static_assert(ndims == 2, "NDIMS error");
    return CpuSliceVarNdArray<CpuSliceVarNdArray<XT>>(std::move(*this), {slice0, slice1});
  }
  CpuSliceVarNdArray<CpuSliceVarNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1,
                                                        Slice&& slice2) {
    static_assert(ndims == 3, "NDIMS error");
    return CpuSliceVarNdArray<CpuSliceVarNdArray<XT>>(std::move(*this), {slice0, slice1, slice2});
  }
  CpuSliceVarNdArray<CpuSliceVarNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1,
                                                        Slice&& slice2, Slice&& slice3) {
    static_assert(ndims == 4, "NDIMS error");
    return CpuSliceVarNdArray<CpuSliceVarNdArray<XT>>(std::move(*this),
                                                      {slice0, slice1, slice2, slice3});
  }
  CpuSliceVarNdArray<CpuSliceVarNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1,
                                                        Slice&& slice2, Slice&& slice3,
                                                        Slice&& slice4) {
    static_assert(ndims == 5, "NDIMS error");
    return CpuSliceVarNdArray<CpuSliceVarNdArray<XT>>(std::move(*this),
                                                      {slice0, slice1, slice2, slice3, slice4});
  }

  template<typename AT>
  void CopyFrom(const AT& ndarray) {
    NdArrayCopy(this, ndarray);
  }

  ALWAYS_INLINE void GetMutPtrAndContiguousSize(int64_t offset, dtype** ptr, size_t* size) const {
    int64_t dim[ndims] = {0};
    this->xpu_shape().template Offset2Coordinate<ndims>(offset, dim);
    for (int i = 0; i < ndims; ++i) { dim[i] = this->slice(i).Get(dim[i]); }
    size_t x_offset = this->x().xpu_shape().template Coordinate2Offset<ndims>(dim);
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
  static std::array<Slice, ndims>&& BoundSlices(const XT& x, std::array<Slice, ndims>&& slices) {
    FOR_RANGE(int32_t, i, 0, ndims) { slices[i].Bound(x.xpu_shape().At(i)); }
    return std::move(slices);
  }
  static Shape BoundedSlices2Shape(const std::array<Slice, ndims>& bounded_slices) {
    std::vector<int64_t> dim_vec;
    for (const Slice& slice : bounded_slices) {
      CHECK_GT(slice.Size(), 0);
      dim_vec.push_back(slice.Size());
    }
    return Shape(dim_vec);
  }
  void SetContiguousLength(const std::array<Slice, ndims>& bounded_slices) {
    contiguous_len_ = 1;
    for (int i = ndims - 1; i >= 0; --i) {
      if (bounded_slices[i].IsContiguous()) { contiguous_len_ *= bounded_slices[i].Size(); }
      if (!(bounded_slices[i].IsContiguous() && bounded_slices[i].IsCoveringAll())) { break; }
    }
  }
  const XT& x_;
  std::array<Slice, ndims> slices_;
  size_t contiguous_len_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_SLICE_NDARRAY_H_
