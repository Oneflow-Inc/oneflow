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
#ifndef ONEFLOW_CORE_COMMON_FLAT_SHAPE_H_
#define ONEFLOW_CORE_COMMON_FLAT_SHAPE_H_

#include <memory>
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow {

class Shape;

// clang-format off

FLAT_MSG_BEGIN(FlatShape);
 public:
  // Methods
  static Maybe<FlatShape> New(const Shape& shape);
  Maybe<void> Init(const Shape& shape);
  Maybe<void> Check(const Shape& shape) const;
  Maybe<void> Check(const FlatShape& flat_shape) const;
  Maybe<Shape> ToShape() const;
  Maybe<void> ToShape(Shape* shape) const;
  int64_t At(int i) const { return dim(i); }
  int64_t NumAxes() const { return dim_size(); }

  // Fields
  FLAT_MSG_DEFINE_REPEATED(int64_t, dim, SHAPE_MAX_AXIS_SIZE);
FLAT_MSG_END(FlatShape);

// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FLAT_SHAPE_H_
