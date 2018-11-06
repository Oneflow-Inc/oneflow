#ifndef ONEFLOW_CORE_COMMON_NDARRAY_H_
#define ONEFLOW_CORE_COMMON_NDARRAY_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T, int NDIMS>
class NdArrayBase {
 protected:
  explicit NdArrayBase(const Shape& shape) : shape_(shape) { TODO(); }
  virtual ~NdArrayBase() = default;

  const Shape& shape() const { return shape_; }
  const std::array<int64_t, NDIMS>& dim_elem_cnt() const { return dim_elem_cnt_; }

  // contiguous buf logically and pysically
  void GetMutPtrAndContiguousSize(int64_t index, T** ptr, size_t* size) const { UNIMPLEMENTED(); }

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

  T At(int64_t dim0) const { UNIMPLEMENTED(); }
};

template<typename T>
class NdArray<T, 2> : public NdArrayBase<T, 2> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 2>(shape) {}
  virtual ~NdArray() = default;

  T At(int64_t dim0, int64_t dim1) const { UNIMPLEMENTED(); }
};

template<typename T>
class NdArray<T, 3> : public NdArrayBase<T, 3> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 3>(shape) {}
  virtual ~NdArray() = default;

  T At(int64_t dim0, int64_t dim1, int64_t dim2) const { UNIMPLEMENTED(); }
};

template<typename T>
class NdArray<T, 4> : public NdArrayBase<T, 4> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 4>(shape) {}
  virtual ~NdArray() = default;

  T At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const { UNIMPLEMENTED(); }
};

template<typename T>
class NdArray<T, 5> : public NdArrayBase<T, 5> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 5>(shape) {}
  virtual ~NdArray() = default;

  T At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const {
    UNIMPLEMENTED();
  }
};

template<typename T, int NDIMS>
class VarNdArray;

template<typename T, int NDIMS>
void AssignVarNdArray(VarNdArray<T, NDIMS>* var_ndarray, const NdArray<T, NDIMS>& ndarray);

template<typename Derived, typename T, int NDIMS>
class VarNdArrayBase : public NdArray<T, NDIMS> {
 public:
  VarNdArrayBase(const Shape& shape, T* ptr)
      : NdArray<T, NDIMS>(shape), ptr_(ptr), len_(shape.elem_cnt()) {}
  virtual ~VarNdArrayBase() = default;

  Derived& operator=(const NdArray<T, 1>& ndarray) {
    auto* derived_this = static_cast<Derived*>(this);
    AssignVarNdArray(derived_this, ndarray);
    return *derived_this;
  }

  void GetMutPtrAndContiguousSize(int64_t index, T** ptr, size_t* size) const override {
    *ptr = ptr_ + index;
    *size = len_ - index;
  }

 private:
  T* ptr_;
  size_t len_;
};

template<typename T>
class VarNdArray<T, 1> final : public VarNdArrayBase<VarNdArray<T, 1>, T, 1> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 1>, T, 1>(shape, ptr) {}
  ~VarNdArray() = default;

  T At(int64_t dim0) const { return ptr_[dim0]; }

 private:
  T* ptr_;
  size_t len_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_NDARRAY_H_
