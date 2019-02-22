#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_H_

#include <climits>
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_shape.h"

namespace oneflow {

template<typename T, int NDIMS>
class NdArrayBase {
 public:
  using dtype = T;
  static const int ndims = NDIMS;
  static const bool immutable = true;

  ALWAYS_INLINE const Shape& shape() const { return shape_; }
  ALWAYS_INLINE const XpuShape& xpu_shape() const { return xpu_shape_; }

 protected:
  explicit NdArrayBase(const Shape& shape) : shape_(shape), xpu_shape_(shape) {
    FOR_RANGE(int, i, 0, NDIMS) { dim_elem_cnt_[i] = shape.Count(i + 1); }
  }
  virtual ~NdArrayBase() = default;

  ALWAYS_INLINE int64_t dim_elem_cnt(int32_t dim) const { return dim_elem_cnt_[dim]; }

 private:
  Shape shape_;
  XpuShape xpu_shape_;
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

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_H_
