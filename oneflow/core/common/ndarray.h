#ifndef ONEFLOW_CORE_COMMON_NDARRAY_H_
#define ONEFLOW_CORE_COMMON_NDARRAY_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T, int NDIMS>
class NdArrayBase {
 protected:
  explicit NdArrayBase(const Shape& shape); 
  virtual ~NdArrayBase() = default;

  const Shape& dim_elem_cnt() const { return shape_; }
  const std::array<int64_t, NDIMS>& dim_elem_cnt() const { return dim_elem_cnt_; }

 private:
  Shape shape_;
  std::array<int64_t, NDIMS> dim_elem_cnt_;
};

template<typename T, int NDIMS> class NdArray;

template<typename T>
class NdArray<T, 1> : public: NdArrayBase<T, 1> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase(shape) { }
  virtual ~NdArray() = default;

  virtual T At(int64_t dim0) const { UNIMPLEMENTED(); }
  virtual const T* Ptr(int64_t dim0) const { UNIMPLEMENTED(); }
  virtual T* MutPtr(int64_t dim0) { UNIMPLEMENTED(); }
  virtual size_t ContiguousLen(int64_t dim0) const { UNIMPLEMENTED(); }
};

template<typename T>
class NdArray<T, 2> : public: NdArrayBase<T, 2> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase(shape) { }
  virtual ~NdArray() = default;

  virtual T At(int64_t dim0, int64_t dim1) const { UNIMPLEMENTED(); }
  virtual const T* Ptr(int64_t dim0, int64_t dim1) const { UNIMPLEMENTED(); }
  virtual T* MutPtr(int64_t dim0, int64_t dim1) { UNIMPLEMENTED(); }
  virtual size_t ContiguousLen(int64_t dim0, int64_t dim1) const { UNIMPLEMENTED(); }
};

template<typename T>
class NdArray<T, 3> : public: NdArrayBase<T, 3> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase(shape) { }
  virtual ~NdArray() = default;

  virtual T At(int64_t dim0, int64_t dim1, int64_t dim2) const { UNIMPLEMENTED(); }
  virtual const T* Ptr(int64_t dim0, int64_t dim1, int64_t dim2) const { UNIMPLEMENTED(); }
  virtual T* MutPtr(int64_t dim0, int64_t dim1, int64_t dim2) { UNIMPLEMENTED(); }
  virtual size_t ContiguousLen(int64_t dim0, int64_t dim1, int64_t dim2) const { UNIMPLEMENTED(); }
};

template<typename T>
class NdArray<T, 4> : public: NdArrayBase<T, 4> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase(shape) { }
  virtual ~NdArray() = default;

  virtual T At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const { UNIMPLEMENTED(); }
  virtual const T* Ptr(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const { UNIMPLEMENTED(); }
  virtual T* MutPtr(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) { UNIMPLEMENTED(); }
  virtual size_t ContiguousLen(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const { UNIMPLEMENTED(); }
};

template<typename T>
class NdArray<T, 5> : public: NdArrayBase<T, 5> {
 public:
  explicit NdArray(const Shape& shape) : NdArrayBase(shape) { }
  virtual ~NdArray() = default;

  virtual T At(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const { UNIMPLEMENTED(); }
  virtual const T* Ptr(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const { UNIMPLEMENTED(); }
  virtual T* MutPtr(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) { UNIMPLEMENTED(); }
  virtual size_t ContiguousLen(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const { UNIMPLEMENTED(); }
};

}

#endif // ONEFLOW_CORE_COMMON_NDARRAY_H_
