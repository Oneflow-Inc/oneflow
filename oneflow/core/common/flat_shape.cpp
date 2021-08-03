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
#include "oneflow/core/common/flat_shape.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

Maybe<void> FlatShape::Init(const std::shared_ptr<const Shape>& shape) {
  CHECK_LE_OR_RETURN(shape->NumAxes(), SHAPE_MAX_AXIS_SIZE);
  this->set_num_axes(shape->NumAxes());
  for (int i = 0; i < this->num_axes(); ++i) { *this->mutable_dim()->Mutable(i) = shape->At(i); }
  return Maybe<void>::Ok();
}

Maybe<void> FlatShape::Check(const std::shared_ptr<const Shape>& shape) const {
  CHECK_EQ_OR_RETURN(this->num_axes(), shape->NumAxes());
  for (int i = 0; i < this->num_axes(); ++i) {
    CHECK_EQ_OR_RETURN(this->dim().Get(i), shape->At(i));
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
