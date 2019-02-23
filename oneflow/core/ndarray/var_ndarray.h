#ifndef ONEFLOW_CORE_NDARRAY_VAR_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_VAR_NDARRAY_H_

#include "oneflow/core/ndarray/ndarray.h"
#include "oneflow/core/ndarray/ndarray_assign.h"

namespace oneflow {

class Slice;
template<typename XT>
class SliceNdArray;

template<typename T, int NDIMS>
class VarNdArray : public NdArray<T, NDIMS> {
 public:
  using dtype = T;
  static const int ndims = NDIMS;
  static const bool immutable = false;
  VarNdArray(const VarNdArray&) = default;
  VarNdArray(const Shape& shape, T* ptr)
      : NdArray<T, NDIMS>(shape), ptr_(ptr), len_(shape.elem_cnt()) {
    CHECK_GT(len_, 0);
  }
  virtual ~VarNdArray() = default;

  SliceNdArray<VarNdArray<T, NDIMS>> operator()(Slice&& slice0) {
    static_assert(NDIMS == 1, "NDIMS error");
    return SliceNdArray<VarNdArray<T, NDIMS>>(std::move(*this), {slice0});
  }
  SliceNdArray<VarNdArray<T, NDIMS>> operator()(Slice&& slice0, Slice&& slice1) {
    static_assert(NDIMS == 2, "NDIMS error");
    return SliceNdArray<VarNdArray<T, NDIMS>>(std::move(*this), {slice0, slice1});
  }
  SliceNdArray<VarNdArray<T, NDIMS>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2) {
    static_assert(NDIMS == 3, "NDIMS error");
    return SliceNdArray<VarNdArray<T, NDIMS>>(std::move(*this), {slice0, slice1, slice2});
  }
  SliceNdArray<VarNdArray<T, NDIMS>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                                Slice&& slice3) {
    static_assert(NDIMS == 4, "NDIMS error");
    return SliceNdArray<VarNdArray<T, NDIMS>>(std::move(*this), {slice0, slice1, slice2, slice3});
  }
  SliceNdArray<VarNdArray<T, NDIMS>> operator()(Slice&& slice0, Slice&& slice1, Slice&& slice2,
                                                Slice&& slice3, Slice&& slice4) {
    static_assert(NDIMS == 5, "NDIMS error");
    return SliceNdArray<VarNdArray<T, NDIMS>>(std::move(*this),
                                              {slice0, slice1, slice2, slice3, slice4});
  }

  template<typename XT>
  void CopyFrom(const XT& ndarray) {
    NdArrayAssign(this, ndarray);
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

#endif  // ONEFLOW_CORE_NDARRAY_VAR_NDARRAY_H_
