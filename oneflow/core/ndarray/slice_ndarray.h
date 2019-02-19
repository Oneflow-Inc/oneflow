#ifndef ONEFLOW_CORE_NDARRAY_SLICE_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_SLICE_NDARRAY_H_

#include "oneflow/core/ndarray/slice.h"
#include "oneflow/core/ndarray/ndarray.h"
#include "oneflow/core/ndarray/ndarray_assign.h"

namespace oneflow {

template<typename Derived, typename XT, int NDIMS>
class SliceNdArrayBase : public NdArray<typename XT::dtype, XT::ndims> {
 public:
  static const bool immutable = XT::immutable;
  static_assert(XT::ndims == NDIMS, "XT::ndims should equals NDIMS");
  SliceNdArrayBase(XT&& x, std::array<Slice, NDIMS>&& slices)
      : NdArray<typename XT::dtype, XT::ndims>(
            BoundedSlices2Shape(BoundSlices(x, std::move(slices)))),
        x_(x),
        slices_(std::move(slices)) {
    SetContiguousLength(slices);
  }
  virtual ~SliceNdArrayBase() = default;

  template<typename AT>
  void Assign(const AT& ndarray) {
    NdArrayAssign(dynamic_cast<Derived*>(this), ndarray);
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
  static std::array<Slice, NDIMS>&& BoundSlices(const XT& x, std::array<Slice, NDIMS>&& slices) {
    FOR_RANGE(int32_t, i, 0, NDIMS) { slices[i].Bound(x.shape().At(i)); }
    return std::move(slices);
  }
  static Shape BoundedSlices2Shape(const std::array<Slice, NDIMS>& bounded_slices) {
    std::vector<int64_t> dim_vec;
    for (const Slice& slice : bounded_slices) {
      CHECK_GT(slice.Size(), 0);
      dim_vec.push_back(slice.Size());
    }
    return Shape(dim_vec);
  }
  void SetContiguousLength(const std::array<Slice, NDIMS>& bounded_slices) {
    contiguous_len_ = 1;
    for (int i = NDIMS - 1; i >= 0; --i) {
      if (bounded_slices[i].IsContiguous()) { contiguous_len_ *= bounded_slices[i].Size(); }
      if (!(bounded_slices[i].IsContiguous() && bounded_slices[i].IsCoveringAll())) { break; }
    }
  }
  const XT& x_;
  std::array<Slice, NDIMS> slices_;
  size_t contiguous_len_;
};

template<typename XT>
class SliceNdArray<XT, typename std::enable_if<XT::ndims == 1>::type> final
    : public SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims> {
 public:
  SliceNdArray(XT&& x, Slice&& slice0)
      : SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims>(std::move(x), {slice0}) {}
  ~SliceNdArray() = default;

  using dtype = typename XT::dtype;
  ALWAYS_INLINE dtype Get(int64_t dim0) const { return this->x().Get(this->slice(0).Get(dim0)); }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable, dtype*>::type Mut(int64_t dim0) const {
    return this->x().Mut(this->slice(0).Get(dim0));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable>::type GetMutPtrAndContiguousSize(
      int64_t offset, dtype** ptr, size_t* size) const {
    size_t dim0 = offset;
    size_t x_offset = this->slice(0).Get(dim0);
    this->GetMutPtrAndMinContiguousSize(offset, x_offset, ptr, size);
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0) {
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), std::move(slice0));
  }
};

template<typename XT>
class SliceNdArray<XT, typename std::enable_if<XT::ndims == 2>::type> final
    : public SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims> {
 public:
  SliceNdArray(XT&& x, Slice&& slice0, Slice&& slice1)
      : SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims>(std::move(x), {slice0, slice1}) {}
  ~SliceNdArray() = default;

  using dtype = typename XT::dtype;
  ALWAYS_INLINE dtype Get(int64_t dim0, int64_t dim1) const {
    return this->x().Get(this->slice(0).Get(dim0), this->slice(1).Get(dim1));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable, dtype*>::type Mut(int64_t dim0,
                                                                          int64_t dim1) const {
    return this->x().Mut(this->slice(0).Get(dim0), this->slice(1).Get(dim1));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable>::type GetMutPtrAndContiguousSize(
      int64_t offset, dtype** ptr, size_t* size) const {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    this->Offset2Dims(offset, &dim0, &dim1);
    size_t x_offset = this->x().Dims2Offset(this->slice(0).Get(dim0), this->slice(1).Get(dim1));
    this->GetMutPtrAndMinContiguousSize(offset, x_offset, ptr, size);
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1) {
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), std::move(slice0), std::move(slice1));
  }
};

template<typename XT>
class SliceNdArray<XT, typename std::enable_if<XT::ndims == 3>::type> final
    : public SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims> {
 public:
  SliceNdArray(XT&& x, Slice&& slice0, Slice&& slice1, Slice&& slice2)
      : SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims>(std::move(x), {slice0, slice1, slice2}) {}
  ~SliceNdArray() = default;

  using dtype = typename XT::dtype;
  ALWAYS_INLINE dtype Get(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return this->x().Get(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable, dtype*>::type Mut(int64_t dim0,
                                                                          int64_t dim1,
                                                                          int64_t dim2) const {
    return this->x().Mut(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable>::type GetMutPtrAndContiguousSize(
      int64_t offset, dtype** ptr, size_t* size) const {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    this->Offset2Dims(offset, &dim0, &dim1, &dim2);
    size_t x_offset = this->x().Dims2Offset(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                                            this->slice(2).Get(dim2));
    this->GetMutPtrAndMinContiguousSize(offset, x_offset, ptr, size);
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2) {
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), std::move(slice0), std::move(slice1),
                                          std::move(slice2));
  }
};

template<typename XT>
class SliceNdArray<XT, typename std::enable_if<XT::ndims == 4>::type> final
    : public SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims> {
 public:
  SliceNdArray(XT&& x, Slice&& slice0, Slice&& slice1, Slice&& slice2, Slice&& slice3)
      : SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims>(std::move(x),
                                                          {slice0, slice1, slice2, slice3}) {}
  ~SliceNdArray() = default;

  using dtype = typename XT::dtype;
  ALWAYS_INLINE dtype Get(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return this->x().Get(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2), this->slice(3).Get(dim3));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable, dtype*>::type Mut(int64_t dim0,
                                                                          int64_t dim1,
                                                                          int64_t dim2,
                                                                          int64_t dim3) const {
    return this->x().Mut(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2), this->slice(3).Get(dim3));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable>::type GetMutPtrAndContiguousSize(
      int64_t offset, dtype** ptr, size_t* size) const {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t dim3 = 0;
    this->Offset2Dims(offset, &dim0, &dim1, &dim2, &dim3);
    size_t x_offset = this->x().Dims2Offset(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                                            this->slice(2).Get(dim2), this->slice(3).Get(dim3));
    this->GetMutPtrAndMinContiguousSize(offset, x_offset, ptr, size);
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                            Slice&& slice3) {
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), std::move(slice0), std::move(slice1),
                                          std::move(slice2), std::move(slice3));
  }
};

template<typename XT>
class SliceNdArray<XT, typename std::enable_if<XT::ndims == 5>::type> final
    : public SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims> {
 public:
  SliceNdArray(XT&& x, Slice&& slice0, Slice&& slice1, Slice&& slice2, Slice&& slice3,
               Slice&& slice4)
      : SliceNdArrayBase<SliceNdArray<XT>, XT, XT::ndims>(
            std::move(x), {slice0, slice1, slice2, slice3, slice4}) {}
  ~SliceNdArray() = default;

  using dtype = typename XT::dtype;
  ALWAYS_INLINE dtype Get(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                          int64_t dim4) const {
    return this->x().Get(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2), this->slice(3).Get(dim3),
                         this->slice(4).Get(dim4));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable, dtype*>::type Mut(
      int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const {
    return this->x().Mut(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2), this->slice(3).Get(dim3),
                         this->slice(4).Get(dim4));
  }
  ALWAYS_INLINE typename std::enable_if<!XT::immutable>::type GetMutPtrAndContiguousSize(
      int64_t offset, dtype** ptr, size_t* size) const {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t dim3 = 0;
    int64_t dim4 = 0;
    this->Offset2Dims(offset, &dim0, &dim1, &dim2, &dim3, &dim4);
    size_t x_offset = this->x().Dims2Offset(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                                            this->slice(2).Get(dim2), this->slice(3).Get(dim3),
                                            this->slice(4).Get(dim4));
    this->GetMutPtrAndMinContiguousSize(offset, x_offset, ptr, size);
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                            Slice&& slice3, Slice&& slice4) {
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), std::move(slice0), std::move(slice1),
                                          std::move(slice2), std::move(slice3), std::move(slice4));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_SLICE_NDARRAY_H_
