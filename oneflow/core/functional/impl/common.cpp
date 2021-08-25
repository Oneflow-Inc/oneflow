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
#include "oneflow/core/functional/impl/common.h"

namespace oneflow {
namespace one {
namespace functional {

bool IsStaticZerosTensor(const std::shared_ptr<Tensor>& x) {
  return nullptr != std::dynamic_pointer_cast<StaticZerosTensor>(x);
}

bool IsInplaceValid(const std::shared_ptr<Tensor>& x) {
  return !(x->is_leaf() && x->requires_grad());
}

Maybe<void> CheckInplaceValid(const std::shared_ptr<Tensor>& x) {
  CHECK_OR_RETURN(IsInplaceValid(x))
      << "a leaf Tensor that requires grad is being used in an in-place operation.";
  return Maybe<void>::Ok();
}

bool IsShapeCanExpandTo(const Shape& shape, const Shape& expand_shape) {
  if (shape == expand_shape) { return true; }
  if (expand_shape.NumAxes() < shape.NumAxes()) { return false; }
  int shift = expand_shape.NumAxes() - shape.NumAxes();
  for (int i = expand_shape.NumAxes() - 1; i >= 0; --i) {
    int index = i - shift;
    if (index >= 0) {
      int dim_a = expand_shape.At(i);
      int dim_b = shape.At(index);
      if (dim_a != dim_b && (dim_a <= 0 || dim_b != 1)) { return false; }
    } else {
      if (expand_shape.At(i) <= 0) { return false; }
    }
  }
  return true;
}

Maybe<void> CheckShapeCanExpandTo(const Shape& shape, const Shape& expand_shape) {
  CHECK_OR_RETURN(IsShapeCanExpandTo(shape, expand_shape))
      << "Can not expand shape " << shape.ToString() << " to " << expand_shape.ToString();
  return Maybe<void>::Ok();
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
