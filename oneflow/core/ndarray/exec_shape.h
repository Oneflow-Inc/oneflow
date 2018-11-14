#ifndef ONEFLOW_CORE_NDARRAY_EXEC_SHAPE_H_
#define ONEFLOW_CORE_NDARRAY_EXEC_SHAPE_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

class ExecShape final {
 public:
  OF_DEVICE_FUNC ExecShape(const ExecShape&) = default;
  OF_DEVICE_FUNC ExecShape(const Shape& shape);

  OF_DEVICE_FUNC int64_t At(int64_t dim) const { return dim_[dim]; }

  OF_DEVICE_FUNC size_t ElemNum() const {
    size_t elem_num = 1;
    for (int i = 0; i < num_axes_; ++i) { elem_num *= dim_[i]; }
    return elem_num;
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
  size_t num_axes_;
  int64_t dim_[OF_PP_SEQ_SIZE(DIM_SEQ)];
  int64_t dim_elem_num_[OF_PP_SEQ_SIZE(DIM_SEQ)];
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_EXEC_SHAPE_H_
