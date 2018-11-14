#include "oneflow/core/ndarray/exec_shape.h"

namespace oneflow {

ExecShape::ExecShape(const Shape& shape) {
  num_axes_ = shape.NumAxes();
  int i = 0;
  for (; i < num_axes_; ++i) {
    dim_[i] = shape.At(i);
    dim_elem_num_[i] = shape.Count(i + 1);
  }
  for (; i < sizeof(dim_) / sizeof(dim_[0]); ++i) {
    dim_[i] = 1;
    dim_elem_num_[i] = 1;
  }
}

}  // namespace oneflow
