#ifndef ONEFLOW_CORE_NDARRAY_VAR_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_VAR_NDARRAY_H_

#include "oneflow/core/ndarray/ndarray.h"
#include "oneflow/core/ndarray/ndarray_assign.h"

namespace oneflow {

template<typename Derived, typename T, int NDIMS>
class VarNdArrayBase : public NdArray<T, NDIMS> {
 public:
  static const bool immutable = false;
  VarNdArrayBase(const VarNdArrayBase&) = default;
  VarNdArrayBase(const Shape& shape, T* ptr)
      : NdArray<T, NDIMS>(shape), ptr_(ptr), len_(shape.elem_cnt()) {
    CHECK_GT(len_, 0);
  }
  virtual ~VarNdArrayBase() = default;

  template<typename XT>
  void CopyFrom(const XT& ndarray) {
    NdArrayAssign(dynamic_cast<Derived*>(this), ndarray);
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

class Slice;
template<typename XT, typename Enable = void>
class SliceNdArray;

template<typename T, int NDIMS>
class VarNdArray;

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

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_VAR_NDARRAY_H_
