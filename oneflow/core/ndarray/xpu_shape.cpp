/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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

XpuShape::XpuShape(const ShapeView& shape) {
  num_axes_ = shape.NumAxes();
  int i = 0;
  for (; i < num_axes_; ++i) { dim_[i] = shape.At(i); }
  UpdateDimElemNumAndElemNum();
  for (; i < sizeof(dim_) / sizeof(dim_[0]); ++i) {
    dim_[i] = 1;
    dim_elem_num_[i] = 1;
  }
}

XpuShape::XpuShape(const ShapeView& shape, int ndims_left_extend_to) {
  if (shape.NumAxes() == 1 && ndims_left_extend_to == 0) {
    num_axes_ = 0;
    int i = 0;
    dim_[i] = shape.At(i);
    UpdateDimElemNumAndElemNum();
    for (; i < sizeof(dim_) / sizeof(dim_[0]); ++i) { dim_[i] = 1; }
  } else {
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

void SimplifyBroadcastShapes(const XpuShape& y, const XpuShape& b, DimVector* simplified_y,
                             DimVector* simplified_b) {
  DimVector simplified_a;
  SimplifyBroadcastShapes(y, y, b, simplified_y, &simplified_a, simplified_b);
}

void SimplifyBroadcastShapes(const XpuShape& y, const XpuShape& a, const XpuShape& b,
                             DimVector* simplified_y, DimVector* simplified_a,
                             DimVector* simplified_b) {
  CHECK_EQ(y.NumAxes(), a.NumAxes());
  CHECK_EQ(b.NumAxes(), a.NumAxes());
  CHECK(simplified_y->empty());
  CHECK(simplified_a->empty());
  CHECK(simplified_b->empty());
  simplified_y->emplace_back(y.At(0));
  simplified_a->emplace_back(a.At(0));
  simplified_b->emplace_back(b.At(0));
  bool a_prev_axis_is_broadcast = (a.At(0) == 1);
  bool b_prev_axis_is_broadcast = (b.At(0) == 1);
  FOR_RANGE(int, i, 1, y.NumAxes()) {
    const int64_t y_dim = y.At(i);
    const int64_t a_dim = a.At(i);
    const int64_t b_dim = b.At(i);
    if ((a_dim == 1) && (b_dim == 1)) { continue; }
    const bool a_cur_axis_is_broadcast = (a_dim == 1);
    const bool b_cur_axis_is_broadcast = (b_dim == 1);
    if (a_prev_axis_is_broadcast == a_cur_axis_is_broadcast
        && b_prev_axis_is_broadcast == b_cur_axis_is_broadcast) {
      simplified_y->back() *= y_dim;
      simplified_a->back() *= a_dim;
      simplified_b->back() *= b_dim;
    } else {
      simplified_y->emplace_back(y_dim);
      simplified_a->emplace_back(a_dim);
      simplified_b->emplace_back(b_dim);
    }
    a_prev_axis_is_broadcast = a_cur_axis_is_broadcast;
    b_prev_axis_is_broadcast = b_cur_axis_is_broadcast;
  }
}

}  // namespace oneflow
