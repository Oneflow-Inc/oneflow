#ifndef ONEFLOW_CORE_COMMON_NDARRAY_H_
#define ONEFLOW_CORE_COMMON_NDARRAY_H_

#include <climits>
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define ALWAYS_INLINE inline

template<typename T, int NDIMS>
class NdArrayBase {
 public:
  using dtype = T;
  static const int ndims = NDIMS;
  static const bool immutable = true;

  ALWAYS_INLINE const Shape& shape() const { return shape_; }

 protected:
  explicit NdArrayBase(const Shape& shape) : shape_(shape) {
    FOR_RANGE(int, i, 0, NDIMS) { dim_elem_cnt_[i] = shape.Count(i + 1); }
  }
  virtual ~NdArrayBase() = default;

  ALWAYS_INLINE int64_t dim_elem_cnt(int32_t dim) const { return dim_elem_cnt_[dim]; }

 private:
  Shape shape_;
  std::array<int64_t, NDIMS> dim_elem_cnt_;
};

template<typename T, int NDIMS>
class NdArray;

template<typename T>
class NdArray<T, 1> : public NdArrayBase<T, 1> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 1>(shape) {}
  virtual ~NdArray() = default;

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0) const { return dim0; }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0) const { *dim0 = offset; }
};

template<typename T>
class NdArray<T, 2> : public NdArrayBase<T, 2> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 2>(shape) {}
  virtual ~NdArray() = default;

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0, int64_t dim1) const {
    return dim0 * this->dim_elem_cnt(0) + dim1;
  }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1) const {
    *dim0 = offset / this->dim_elem_cnt(0);
    *dim1 = offset % this->dim_elem_cnt(0);
  }
};

template<typename T>
class NdArray<T, 3> : public NdArrayBase<T, 3> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 3>(shape) {}
  virtual ~NdArray() = default;

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return dim0 * this->dim_elem_cnt(0) + dim1 * this->dim_elem_cnt(1) + dim2;
  }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1,
                                 int64_t* dim2) const {
    *dim0 = offset / this->dim_elem_cnt(0);
    offset = offset % this->dim_elem_cnt(0);
    *dim1 = offset / this->dim_elem_cnt(1);
    *dim2 = offset % this->dim_elem_cnt(1);
  }
};

template<typename T>
class NdArray<T, 4> : public NdArrayBase<T, 4> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 4>(shape) {}
  virtual ~NdArray() = default;

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return dim0 * this->dim_elem_cnt(0) + dim1 * this->dim_elem_cnt(1)
           + dim2 * this->dim_elem_cnt(2) + dim3;
  }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1, int64_t* dim2,
                                 int64_t* dim3) const {
    *dim0 = offset / this->dim_elem_cnt(0);
    offset = offset % this->dim_elem_cnt(0);
    *dim1 = offset / this->dim_elem_cnt(1);
    offset = offset % this->dim_elem_cnt(1);
    *dim2 = offset / this->dim_elem_cnt(2);
    *dim3 = offset % this->dim_elem_cnt(2);
  }
};

template<typename T>
class NdArray<T, 5> : public NdArrayBase<T, 5> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 5>(shape) {}
  virtual ~NdArray() = default;

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                                    int64_t dim4) const {
    return dim0 * this->dim_elem_cnt(0) + dim1 * this->dim_elem_cnt(1)
           + dim2 * this->dim_elem_cnt(2) + dim3 * this->dim_elem_cnt(3) + dim4;
  }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1, int64_t* dim2,
                                 int64_t* dim3, int64_t* dim4) const {
    *dim0 = offset / this->dim_elem_cnt(0);
    offset = offset % this->dim_elem_cnt(0);
    *dim1 = offset / this->dim_elem_cnt(1);
    offset = offset % this->dim_elem_cnt(1);
    *dim2 = offset / this->dim_elem_cnt(2);
    offset = offset % this->dim_elem_cnt(2);
    *dim3 = offset / this->dim_elem_cnt(3);
    *dim4 = offset % this->dim_elem_cnt(3);
  }
};

template<typename T, int NDIMS>
class VarNdArray;

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && !XT::immutable>::type AssignNdArray(YT* y_ndarray,
                                                                              const XT& x_ndarray);
template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 1>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray);
template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 2>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray);
template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 3>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray);
template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 4>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray);
template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 5>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray);

template<typename Derived, typename T, int NDIMS>
class VarNdArrayBase : public NdArray<T, NDIMS> {
 public:
  static const bool immutable = false;
  VarNdArrayBase(const Shape& shape, T* ptr)
      : NdArray<T, NDIMS>(shape), ptr_(ptr), len_(shape.elem_cnt()) {}
  virtual ~VarNdArrayBase() = default;

  template<typename XT>
  void Assign(const XT& ndarray) {
    AssignNdArray(dynamic_cast<Derived*>(this), ndarray);
  }

  ALWAYS_INLINE void GetMutPtrAndContiguousSize(T** ptr, size_t* size, int64_t offset) const {
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

class Slice;
template<typename XT, typename Enable = void>
class SliceNdArray;

template<typename T>
class VarNdArray<T, 1> final : public VarNdArrayBase<VarNdArray<T, 1>, T, 1> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 1>, T, 1>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T Get(int64_t dim0) const { return this->ptr()[dim0]; }
  ALWAYS_INLINE T* Mut(int64_t dim0) const { return this->ptr() + dim0; }
  SliceNdArray<VarNdArray<T, 1>> operator()(Slice&& slice0) {
    return SliceNdArray<VarNdArray<T, 1>>(std::move(*this), std::move(slice0));
  }
};

template<typename T>
class VarNdArray<T, 2> final : public VarNdArrayBase<VarNdArray<T, 2>, T, 2> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 2>, T, 2>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T Get(int64_t dim0, int64_t dim1) const {
    return this->ptr()[this->Dims2Offset(dim0, dim1)];
  }
  ALWAYS_INLINE T* Mut(int64_t dim0, int64_t dim1) const {
    return this->ptr() + this->Dims2Offset(dim0, dim1);
  }
  SliceNdArray<VarNdArray<T, 2>> operator()(Slice&& slice0, Slice&& slice1) {
    return SliceNdArray<VarNdArray<T, 2>>(std::move(*this), std::move(slice0), std::move(slice1));
  }
};

template<typename T>
class VarNdArray<T, 3> final : public VarNdArrayBase<VarNdArray<T, 3>, T, 3> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 3>, T, 3>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T Get(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return this->ptr()[this->Dims2Offset(dim0, dim1, dim2)];
  }
  ALWAYS_INLINE T* Mut(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return this->ptr() + this->Dims2Offset(dim0, dim1, dim2);
  }
  SliceNdArray<VarNdArray<T, 3>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2) {
    return SliceNdArray<VarNdArray<T, 3>>(std::move(*this), std::move(slice0), std::move(slice1),
                                          std::move(slice2));
  }
};

template<typename T>
class VarNdArray<T, 4> final : public VarNdArrayBase<VarNdArray<T, 4>, T, 4> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 4>, T, 4>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T Get(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return *this->Mut(dim0, dim1, dim2, dim3);
  }
  ALWAYS_INLINE T* Mut(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return this->ptr() + this->Dims2Offset(dim0, dim1, dim2, dim3);
  }
  SliceNdArray<VarNdArray<T, 4>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                            Slice&& slice3) {
    return SliceNdArray<VarNdArray<T, 4>>(std::move(*this), std::move(slice0), std::move(slice1),
                                          std::move(slice2), std::move(slice3));
  }
};

template<typename T>
class VarNdArray<T, 5> final : public VarNdArrayBase<VarNdArray<T, 5>, T, 5> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 5>, T, 5>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T Get(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const {
    return this->ptr()[this->Dims2Offset(dim0, dim1, dim2, dim3, dim4)];
  }
  ALWAYS_INLINE T* Mut(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const {
    return this->ptr() + this->Dims2Offset(dim0, dim1, dim2, dim3, dim4);
  }
  SliceNdArray<VarNdArray<T, 5>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                            Slice&& slice3, Slice&& slice4) {
    return SliceNdArray<VarNdArray<T, 5>>(std::move(*this), std::move(slice0), std::move(slice1),
                                          std::move(slice2), std::move(slice3), std::move(slice4));
  }
};

class Slice final {
 public:
  static const int64_t kStart = LLONG_MIN;
  static const int64_t kEnd = LLONG_MAX;

  Slice(const Slice&) = default;
  Slice(int64_t index) : start_(index), end_(index + 1), stride_(1), value_capacity_(0) {}
  Slice(const std::initializer_list<int64_t>& l);
  ~Slice() = default;

  const Slice& Bound(size_t value_capacity);

  ALWAYS_INLINE int64_t Get(int64_t index) const { return start_ + index * stride_; }
  bool IsBounded() const;
  size_t Size() const;
  bool is_contiguous() const;
  bool is_covering_all() const;

 private:
  int64_t start_;
  int64_t end_;
  int64_t stride_;
  size_t value_capacity_;
};

template<typename Derived, typename XT, int NDIMS>
class SliceNdArrayBase : public NdArray<typename XT::dtype, XT::ndims> {
 public:
  static const bool immutable = false;
  static_assert(XT::ndims == NDIMS, "XT::ndims should equals NDIMS");
  static_assert(!XT::immutable, "XT should be mutable");
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
    AssignNdArray(dynamic_cast<Derived*>(this), ndarray);
  }

 protected:
  ALWAYS_INLINE const XT& x() const { return x_; }
  ALWAYS_INLINE const Slice& slice(int32_t dim) const { return slices_[dim]; }
  ALWAYS_INLINE void GetMutPtrAndMinContiguousSize(typename XT::dtype** ptr, size_t* size,
                                                   int64_t offset, int64_t x_offset) const {
    size_t x_contiguous_size;
    this->x().GetMutPtrAndContiguousSize(ptr, &x_contiguous_size, x_offset);
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
      if (bounded_slices[i].is_contiguous()) { contiguous_len_ *= bounded_slices[i].Size(); }
      if (!(bounded_slices[i].is_contiguous() && bounded_slices[i].is_covering_all())) { break; }
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
  ALWAYS_INLINE dtype* Mut(int64_t dim0) const { return this->x().Mut(this->slice(0).Get(dim0)); }
  ALWAYS_INLINE void GetMutPtrAndContiguousSize(dtype** ptr, size_t* size, int64_t offset) const {
    size_t dim0 = offset;
    size_t x_offset = this->slice(0).Get(dim0);
    this->GetMutPtrAndMinContiguousSize(ptr, size, offset, x_offset);
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
  ALWAYS_INLINE dtype* Mut(int64_t dim0, int64_t dim1) const {
    return this->x().Mut(this->slice(0).Get(dim0), this->slice(1).Get(dim1));
  }
  ALWAYS_INLINE void GetMutPtrAndContiguousSize(dtype** ptr, size_t* size, int64_t offset) const {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    this->Offset2Dims(offset, &dim0, &dim1);
    size_t x_offset = this->x().Dims2Offset(this->slice(0).Get(dim0), this->slice(1).Get(dim1));
    this->GetMutPtrAndMinContiguousSize(ptr, size, offset, x_offset);
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
  ALWAYS_INLINE dtype* Mut(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return this->x().Mut(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2));
  }
  ALWAYS_INLINE void GetMutPtrAndContiguousSize(dtype** ptr, size_t* size, int64_t offset) const {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    this->Offset2Dims(offset, &dim0, &dim1, &dim2);
    size_t x_offset = this->x().Dims2Offset(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                                            this->slice(2).Get(dim2));
    this->GetMutPtrAndMinContiguousSize(ptr, size, offset, x_offset);
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
  ALWAYS_INLINE dtype* Mut(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return this->x().Mut(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2), this->slice(3).Get(dim3));
  }
  ALWAYS_INLINE void GetMutPtrAndContiguousSize(dtype** ptr, size_t* size, int64_t offset) const {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t dim3 = 0;
    this->Offset2Dims(offset, &dim0, &dim1, &dim2, &dim3);
    size_t x_offset = this->x().Dims2Offset(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                                            this->slice(2).Get(dim2), this->slice(3).Get(dim3));
    this->GetMutPtrAndMinContiguousSize(ptr, size, offset, x_offset);
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
  ALWAYS_INLINE dtype* Mut(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                           int64_t dim4) const {
    return this->x().Mut(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                         this->slice(2).Get(dim2), this->slice(3).Get(dim3),
                         this->slice(4).Get(dim4));
  }
  ALWAYS_INLINE void GetMutPtrAndContiguousSize(dtype** ptr, size_t* size, int64_t offset) const {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t dim3 = 0;
    int64_t dim4 = 0;
    this->Offset2Dims(offset, &dim0, &dim1, &dim2, &dim3, &dim4);
    size_t x_offset = this->x().Dims2Offset(this->slice(0).Get(dim0), this->slice(1).Get(dim1),
                                            this->slice(2).Get(dim2), this->slice(3).Get(dim3),
                                            this->slice(4).Get(dim4));
    this->GetMutPtrAndMinContiguousSize(ptr, size, offset, x_offset);
  }
  SliceNdArray<SliceNdArray<XT>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                            Slice&& slice3, Slice&& slice4) {
    return SliceNdArray<SliceNdArray<XT>>(std::move(*this), std::move(slice0), std::move(slice1),
                                          std::move(slice2), std::move(slice3), std::move(slice4));
  }
};

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && !XT::immutable>::type AssignNdArray(YT* y_ndarray,
                                                                              const XT& x_ndarray) {
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  T* dst_ptr = nullptr;
  size_t dst_size = 0;
  T* src_ptr = nullptr;
  size_t src_size = 0;
  int64_t cur_index = 0;
  size_t total_elem_cnt = y_ndarray->shape().elem_cnt();
  while (cur_index < total_elem_cnt) {
    if (dst_size == 0) { y_ndarray->GetMutPtrAndContiguousSize(&dst_ptr, &dst_size, cur_index); }
    if (src_size == 0) { x_ndarray.GetMutPtrAndContiguousSize(&src_ptr, &src_size, cur_index); }
    if (src_size == 0) { break; }
    size_t cp_size = std::min(dst_size, src_size);
    if (cp_size == 1) {
      *dst_ptr = *src_ptr;
    } else {
      memcpy(dst_ptr, src_ptr, sizeof(T) * cp_size);
    }
    dst_ptr += cp_size;
    src_ptr += cp_size;
    dst_size -= cp_size;
    src_size -= cp_size;
    cur_index += cp_size;
  }
  CHECK_EQ(dst_size, 0);
  CHECK_EQ(src_size, 0);
  CHECK_EQ(cur_index, total_elem_cnt);
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 1>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 1);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  FOR_RANGE(int64_t, i, 0, dim0_size) { *y_ndarray->Mut(i) = x_ndarray.Get(i); }
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 2>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 2);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  int64_t dim1_size = y_ndarray->shape().At(1);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    FOR_RANGE(int64_t, j, 0, dim1_size) { *y_ndarray->Mut(i, j) = x_ndarray.Get(i, j); }
  }
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 3>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 3);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  int64_t dim1_size = y_ndarray->shape().At(1);
  int64_t dim2_size = y_ndarray->shape().At(2);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      FOR_RANGE(int64_t, k, 0, dim2_size) { *y_ndarray->Mut(i, j, k) = x_ndarray.Get(i, j, k); }
    }
  }
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 4>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 4);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  int64_t dim1_size = y_ndarray->shape().At(1);
  int64_t dim2_size = y_ndarray->shape().At(2);
  int64_t dim3_size = y_ndarray->shape().At(3);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      FOR_RANGE(int64_t, k, 0, dim2_size) {
        FOR_RANGE(int64_t, n, 0, dim3_size) {
          *y_ndarray->Mut(i, j, k, n) = x_ndarray.Get(i, j, k, n);
        }
      }
    }
  }
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 5>::type AssignNdArray(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 5);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  int64_t dim1_size = y_ndarray->shape().At(1);
  int64_t dim2_size = y_ndarray->shape().At(2);
  int64_t dim3_size = y_ndarray->shape().At(3);
  int64_t dim4_size = y_ndarray->shape().At(4);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      FOR_RANGE(int64_t, k, 0, dim2_size) {
        FOR_RANGE(int64_t, n, 0, dim3_size) {
          FOR_RANGE(int64_t, m, 0, dim4_size) {
            *y_ndarray->Mut(i, j, k, n, m) = x_ndarray.Get(i, j, k, n, m);
          }
        }
      }
    }
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_NDARRAY_H_
