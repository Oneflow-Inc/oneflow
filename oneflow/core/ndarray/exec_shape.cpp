#include "oneflow/core/ndarray/exec_shape.h"

namespace oneflow {

ExecShape::ExecShape(const Shape& shape) {
  num_axes_ = shape.NumAxes();
  int i = 0;
  for (; i < num_axes_; ++i) { dim_[i] = shape.At(i); }
  UpdateDimElemNumAndElemNum();
  for (; i < sizeof(dim_) / sizeof(dim_[0]); ++i) {
    dim_[i] = 1;
    dim_elem_num_[i] = 1;
  }
}

bool ExecShape::operator==(const ExecShape& rhs) const {
  if (num_axes_ != rhs.num_axes_) { return false; }
  if (elem_num_ != rhs.elem_num_) { return false; }
  for (int i = 0; i < num_axes_; ++i) {
    if (dim_[i] != rhs.dim_[i]) { return false; }
    if (dim_elem_num_[i] != rhs.dim_elem_num_[i]) { return false; }
  }
  return true;
}

}  // namespace oneflow
