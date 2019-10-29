#include "oneflow/core/ndarray/xpu_shape.h"

namespace oneflow {

XpuShape::XpuShape(const int64_t dim[], int num_axes) {
  num_axes_ = num_axes;
  int i = 0;
  for (; i < num_axes_; ++i) { dim_[i] = dim[i]; }
  UpdateDimElemNumAndElemNum();
  for (; i < sizeof(dim_) / sizeof(dim_[0]); ++i) {
    dim_[i] = 1;
    dim_elem_num_[i] = 1;
  }
}

XpuShape::XpuShape(const Shape& shape) {
  num_axes_ = shape.NumAxes();
  int i = 0;
  for (; i < num_axes_; ++i) { dim_[i] = shape.At(i); }
  UpdateDimElemNumAndElemNum();
  for (; i < sizeof(dim_) / sizeof(dim_[0]); ++i) {
    dim_[i] = 1;
    dim_elem_num_[i] = 1;
  }
}

XpuShape::XpuShape(const DenseShapeView& shape) {
  num_axes_ = shape.NumAxes();
  int i = 0;
  for (; i < num_axes_; ++i) { dim_[i] = shape.At(i); }
  UpdateDimElemNumAndElemNum();
  for (; i < sizeof(dim_) / sizeof(dim_[0]); ++i) {
    dim_[i] = 1;
    dim_elem_num_[i] = 1;
  }
}

XpuShape::XpuShape(const DenseShapeView& shape, int ndims_left_extend_to) {
  CHECK_LE(shape.NumAxes(), ndims_left_extend_to);
  num_axes_ = ndims_left_extend_to;
  size_t left_ones_num = num_axes_ - shape.NumAxes();
  int i = 0;
  for (; i < left_ones_num; ++i) { dim_[i] = 1; }
  for (; i < num_axes_; ++i) { dim_[i] = shape.At(i - left_ones_num); }
  UpdateDimElemNumAndElemNum();
  for (; i < sizeof(dim_) / sizeof(dim_[0]); ++i) {
    dim_[i] = 1;
    dim_elem_num_[i] = 1;
  }
}

bool XpuShape::operator==(const XpuShape& rhs) const {
  if (num_axes_ != rhs.num_axes_) { return false; }
  if (elem_num_ != rhs.elem_num_) { return false; }
  for (int i = 0; i < num_axes_; ++i) {
    if (dim_[i] != rhs.dim_[i]) { return false; }
    if (dim_elem_num_[i] != rhs.dim_elem_num_[i]) { return false; }
  }
  return true;
}

}  // namespace oneflow
