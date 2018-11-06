#ifndef ONEFLOW_CORE_COMMON_NDARRAY_H_
#define ONEFLOW_CORE_COMMON_NDARRAY_H_

#include <climits>
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define ALWAYS_INLINE inline

template<typename T, int NDIMS>
class NdArrayBase {
 protected:
  explicit NdArrayBase(const Shape& shape) : shape_(shape) { TODO(); }
  virtual ~NdArrayBase() = default;

  ALWAYS_INLINE const Shape& shape() const { return shape_; }
  ALWAYS_INLINE int64_t dim_elem_cnt(int32_t dim) const { return dim_elem_cnt_[dim]; }

  // contiguous buf logically and pysically
  virtual void GetMutPtrAndContiguousSize(int64_t index, T** ptr, size_t* size) const {
    UNIMPLEMENTED();
  }

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

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0) { return dim0; }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0) { *dim0 = offset; }
};

template<typename T>
class NdArray<T, 2> : public NdArrayBase<T, 2> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 2>(shape) {}
  virtual ~NdArray() = default;

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0, int64_t dim1) {
    return dim0 * this->dim_elem_cnt(0) + dim1;
  }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1) {
    *dim0 = offset / this->dim_elem_cnt(0);
    *dim1 = offset % this->dim_elem_cnt(0);
  }
};

template<typename T>
class NdArray<T, 3> : public NdArrayBase<T, 3> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase<T, 3>(shape) {}
  virtual ~NdArray() = default;

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0, int64_t dim1, int64_t dim2) {
    return dim0 * this->dim_elem_cnt(0) + dim1 * this->dim_elem_cnt(1) + dim2;
  }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1, int64_t* dim2) {
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

  ALWAYS_INLINE int64_t Dims2Offset(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) {
    return dim0 * this->dim_elem_cnt(0) + dim1 * this->dim_elem_cnt(1)
           + dim2 * this->dim_elem_cnt(2) + dim3;
  }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1, int64_t* dim2,
                                 int64_t* dim3) {
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
                                    int64_t dim4) {
    return dim0 * this->dim_elem_cnt(0) + dim1 * this->dim_elem_cnt(1)
           + dim2 * this->dim_elem_cnt(2) + dim3 * this->dim_elem_cnt(3) + dim4;
  }
  ALWAYS_INLINE void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1, int64_t* dim2,
                                 int64_t* dim3, int64_t* dim4) {
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

  virtual void GetMutPtrAndContiguousSize(int64_t index, T** ptr, size_t* size) const override {
    *ptr = ptr_ + index;
    *size = len_ - index;
  }

 protected:
  const T* ptr() const { return ptr_; }

 private:
  T* ptr_;
  size_t len_;
};

template<typename T>
class VarNdArray<T, 1> final : public VarNdArrayBase<VarNdArray<T, 1>, T, 1> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 1>, T, 1>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T At(int64_t dim0) const { return this->ptr()[dim0]; }
};

template<typename T>
class VarNdArray<T, 2> final : public VarNdArrayBase<VarNdArray<T, 2>, T, 2> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 2>, T, 2>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T At(int64_t dim0, int64_t dim1) const {
    return *(this->ptr() + this->Dims2Offset(dim0, dim1));
  }
};

template<typename T>
class VarNdArray<T, 3> final : public VarNdArrayBase<VarNdArray<T, 3>, T, 3> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 3>, T, 3>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T At(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return *(this->ptr() + this->Dims2Offset(dim0, dim1, dim2));
  }
};

template<typename T>
class VarNdArray<T, 4> final : public VarNdArrayBase<VarNdArray<T, 4>, T, 4> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 4>, T, 4>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return *(this->ptr() + this->Dims2Offset(dim0, dim1, dim2, dim3));
  }
};

template<typename T>
class VarNdArray<T, 5> final : public VarNdArrayBase<VarNdArray<T, 5>, T, 5> {
 public:
  VarNdArray(const Shape& shape, T* ptr) : VarNdArrayBase<VarNdArray<T, 5>, T, 5>(shape, ptr) {}
  ~VarNdArray() = default;

  ALWAYS_INLINE T At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const {
    return *(this->ptr() + this->Dims2Offset(dim0, dim1, dim2, dim3, dim4));
  }
};

class Slice final {
 public:
  static const int64_t kStart = LLONG_MIN;
  static const int64_t kEnd = LLONG_MAX;
  Slice(int64_t index) : start_(index), end_(index + 1), stride_(1), value_capacity_(0) {}
  Slice(const std::initializer_list<int64_t>& l);

  void Bound(size_t value_capacity);

  int64_t At(int64_t index) const;
  bool IsBounded() const;
  size_t Size() const;
  bool is_contiguous() const;

 private:
  int64_t start_;
  int64_t end_;
  int64_t stride_;
  size_t value_capacity_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_NDARRAY_H_
