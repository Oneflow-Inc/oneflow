#ifndef ONEFLOW_CORE_NDARRAY_SLICE_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_SLICE_NDARRAY_H_

#include "oneflow/core/ndarray/slice.h"
#include "oneflow/core/ndarray/ndarray.h"
#include "oneflow/core/ndarray/ndarray_assign.h"

namespace oneflow {

template<typename XT>
class SliceNdArray : public NdArray<typename XT::dtype, XT::ndims> {
 public:
  static const int ndims = XT::ndims;
  static const bool immutable = XT::immutable;
  SliceNdArray(XT&& x, std::array<Slice, ndims>&& slices)
      : NdArray<typename XT::dtype, XT::ndims>(
            BoundedSlices2Shape(BoundSlices(x, std::move(slices)))),
        x_(x),
        slices_(std::move(slices)) {
    SetContiguousLength(slices);
  }
  virtual ~SliceNdArray() = default;

  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0) {
    static_assert(ndims == 1, "NDIMS error");
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), {slice0});
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1) {
    static_assert(ndims == 2, "NDIMS error");
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), {slice0, slice1});
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2) {
    static_assert(ndims == 3, "NDIMS error");
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), {slice0, slice1, slice2});
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                            Slice&& slice3) {
    static_assert(ndims == 4, "NDIMS error");
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), {slice0, slice1, slice2, slice3});
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                            Slice&& slice3, Slice&& slice4) {
    static_assert(ndims == 5, "NDIMS error");
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this),
                                          {slice0, slice1, slice2, slice3, slice4});
  }

  template<typename AT>
  void CopyFrom(const AT& ndarray) {
    NdArrayAssign(this, ndarray);
  }

  using dtype = typename XT::dtype;
  ALWAYS_INLINE typename std::enable_if<!XT::immutable>::type GetMutPtrAndContiguousSize(
      int64_t offset, dtype** ptr, size_t* size) const {
    int64_t dim[ndims] = {0};
    this->xpu_shape().template Offset2Coordinate<ndims>(offset, dim);
    for (int i = 0; i < ndims; ++i) { dim[i] = this->slice(i).Get(dim[i]); }
    size_t x_offset = this->x().xpu_shape().template Coordinate2Offset<ndims>(dim);
    this->GetMutPtrAndMinContiguousSize(offset, x_offset, ptr, size);
  }

 protected:
  ALWAYS_INLINE const XT& x() const { return x_; }
  ALWAYS_INLINE const Slice& slice(int32_t dim) const { return slices_[dim]; }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable>::type GetMutPtrAndMinContiguousSize(
      int64_t offset, int64_t x_offset, typename XT::dtype** ptr, size_t* size) const {
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
