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
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

void ShapeView::ToDimVector(DimVector* dim_vec) const {
  dim_vec->resize(this->size());
  dim_vec->assign(this->data(), this->data() + this->size());
}

void ShapeView::ToShape(Shape* shape) const {
  DimVector dim_vec;
  this->ToDimVector(&dim_vec);
  *shape = Shape(dim_vec);
}

std::ostream& operator<<(std::ostream& out, ShapeView shape) {
  out << shape.ToString();
  return out;
}

void MutShapeView::set_shape(ShapeView shape) {
  if (shape.ptr() == mut_ptr() && shape.NumAxes() == NumAxes()) { return; }
  CHECK_EQ(NumAxes(), shape.NumAxes());
  std::copy(shape.ptr(), shape.ptr() + shape.NumAxes(), mut_ptr());
}

}  // namespace oneflow
