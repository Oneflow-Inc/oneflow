#ifndef ONEFLOW_CORE_NDARRAY_EXEC_SHAPE_H_
#define ONEFLOW_CORE_NDARRAY_EXEC_SHAPE_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

class ExecShape final {
 public:
  explicit ExecShape(const Shape& shape);
  OF_DEVICE_FUNC ExecShape(const int64_t dim[], int num_axes);
  OF_DEVICE_FUNC ExecShape(const ExecShape&) = default;

  OF_DEVICE_FUNC int64_t At(int64_t dim) const { return dim_[dim]; }

  OF_DEVICE_FUNC size_t ElemNum() const { return elem_num_; }
  OF_DEVICE_FUNC size_t NumAxes() const { return num_axes_; }
  size_t HostElemNum() const { return elem_num_; }
  bool operator==(const ExecShape&) const;

  OF_DEVICE_FUNC void Set(int64_t axis, int64_t value) {
    dim_[axis] = value;
    UpdateDimElemNumAndElemNum();
  }

  OF_DEVICE_FUNC int64_t Dims2Offset(int64_t dim0) const { return dim0; }

  OF_DEVICE_FUNC int64_t Dims2Offset(int64_t dim0, int64_t dim1) const {
    return dim0 * dim_elem_num_[0] + dim1;
  }

  OF_DEVICE_FUNC int64_t Dims2Offset(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return dim0 * dim_elem_num_[0] + dim1 * dim_elem_num_[1] + dim2;
  }
  OF_DEVICE_FUNC int64_t Dims2Offset(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return dim0 * dim_elem_num_[0] + dim1 * dim_elem_num_[1] + dim2 * dim_elem_num_[2] + dim3;
  }
  OF_DEVICE_FUNC int64_t Dims2Offset(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                                     int64_t dim4) const {
    return dim0 * dim_elem_num_[0] + dim1 * dim_elem_num_[1] + dim2 * dim_elem_num_[2]
           + dim3 * dim_elem_num_[3] + dim4;
  }

  OF_DEVICE_FUNC void Offset2Dims(int64_t offset, int64_t* dim0) const { *dim0 = offset; }
  OF_DEVICE_FUNC void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1) const {
    *dim0 = offset / dim_elem_num_[0];
    *dim1 = offset % dim_elem_num_[0];
  }
  OF_DEVICE_FUNC void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1,
                                  int64_t* dim2) const {
    *dim0 = offset / dim_elem_num_[0];
    offset = offset % dim_elem_num_[0];
    *dim1 = offset / dim_elem_num_[1];
    *dim2 = offset % dim_elem_num_[1];
  }
  OF_DEVICE_FUNC void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1, int64_t* dim2,
                                  int64_t* dim3) const {
    *dim0 = offset / dim_elem_num_[0];
    offset = offset % dim_elem_num_[0];
    *dim1 = offset / dim_elem_num_[1];
    offset = offset % dim_elem_num_[1];
    *dim2 = offset / dim_elem_num_[2];
    *dim3 = offset % dim_elem_num_[2];
  }
  OF_DEVICE_FUNC void Offset2Dims(int64_t offset, int64_t* dim0, int64_t* dim1, int64_t* dim2,
                                  int64_t* dim3, int64_t* dim4) const {
    *dim0 = offset / dim_elem_num_[0];
    offset = offset % dim_elem_num_[0];
    *dim1 = offset / dim_elem_num_[1];
    offset = offset % dim_elem_num_[1];
    *dim2 = offset / dim_elem_num_[2];
    offset = offset % dim_elem_num_[2];
    *dim3 = offset / dim_elem_num_[3];
    *dim4 = offset % dim_elem_num_[3];
  }

 private:
  OF_DEVICE_FUNC void UpdateDimElemNumAndElemNum() {
    elem_num_ = 1;
    for (int i = num_axes_ - 1; i >= 0; --i) {
      dim_elem_num_[i] = elem_num_;
      elem_num_ *= dim_[i];
    }
  }

  size_t num_axes_;
  size_t elem_num_;
  int64_t dim_[OF_PP_SEQ_SIZE(DIM_SEQ)];
  int64_t dim_elem_num_[OF_PP_SEQ_SIZE(DIM_SEQ)];
};

template<int NDIMS>
struct ExecShapeUtil;

template<>
struct ExecShapeUtil<1> final {
  OF_DEVICE_FUNC static int64_t DimVec2Offset(const ExecShape& shape, int64_t dim[]) {
    return shape.Dims2Offset(dim[0]);
  }
  OF_DEVICE_FUNC static void Offset2DimVec(const ExecShape& shape, int64_t offset, int64_t dim[]) {
    shape.Offset2Dims(offset, &dim[0]);
  }
};

#define PARAM_DIM_AND_COMMA(i) dim[i],
#define PARAM_REF_DIM_AND_COMMA(i) &dim[i],
#define SPECIALIZE_EXEC_SHAPE_UTIL(n)                                                             \
  template<>                                                                                      \
  struct ExecShapeUtil<n + 2> final {                                                             \
    OF_DEVICE_FUNC static int64_t DimVec2Offset(const ExecShape& shape, int64_t dim[]) {          \
      return shape.Dims2Offset(OF_PP_FOR_EACH_TUPLE(PARAM_DIM_AND_COMMA, GET_SEQ(n)) dim[n + 1]); \
    }                                                                                             \
    OF_DEVICE_FUNC static void Offset2DimVec(const ExecShape& shape, int64_t offset,              \
                                             int64_t dim[]) {                                     \
      return shape.Offset2Dims(                                                                   \
          offset, OF_PP_FOR_EACH_TUPLE(PARAM_REF_DIM_AND_COMMA, GET_SEQ(n)) & dim[n + 1]);        \
    }                                                                                             \
  };

SPECIALIZE_EXEC_SHAPE_UTIL(0);
SPECIALIZE_EXEC_SHAPE_UTIL(1);
SPECIALIZE_EXEC_SHAPE_UTIL(2);
SPECIALIZE_EXEC_SHAPE_UTIL(3);
#undef SPECIALIZE_EXEC_SHAPE_UTIL
#undef PARAM_DIM_AND_COMMA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_EXEC_SHAPE_H_
