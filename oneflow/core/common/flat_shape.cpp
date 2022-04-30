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

/*static*/ Maybe<FlatShape> FlatShape::New(const Shape& shape) {
  const auto& flat_shape = std::make_shared<FlatShape>();
  JUST(flat_shape->Init(shape));
  return flat_shape;
}

Maybe<void> FlatShape::Init(const Shape& shape) {
  CHECK_LE_OR_RETURN(shape.NumAxes(), SHAPE_MAX_AXIS_SIZE);
  this->clear_dim();
  for (int i = 0; i < shape.NumAxes(); ++i) { *this->mutable_dim()->Add() = shape.At(i); }
  return Maybe<void>::Ok();
}

Maybe<void> FlatShape::Check(const Shape& shape) const {
  CHECK_EQ_OR_RETURN(this->dim_size(), shape.NumAxes())
      << Error::RuntimeError()
      << "Expected same shape on each rank, but found at least two shapes, "
      << JUST(ToShape())->ToString() << " and " << shape.ToString() << "!";
  for (int i = 0; i < this->dim_size(); ++i) { CHECK_EQ_OR_RETURN(this->dim(i), shape.At(i)); }
  return Maybe<void>::Ok();
}

Maybe<void> FlatShape::Check(const FlatShape& flat_shape) const {
  CHECK_EQ_OR_RETURN(this->dim_size(), flat_shape.NumAxes())
      << Error::RuntimeError()
      << "Expected input of each rank must have the same size, but got at least two size, "
      << JUST(ToShape())->ToString() << " and " << JUST(flat_shape.ToShape())->ToString();
  for (int i = 0; i < this->dim_size(); ++i) {
    CHECK_EQ_OR_RETURN(this->dim(i), flat_shape.At(i))
        << Error::RuntimeError()
        << "Expected input of each rank must have the same size, but got at least two size, "
        << JUST(ToShape())->ToString() << " and " << JUST(flat_shape.ToShape())->ToString();
  }
  return Maybe<void>::Ok();
}

Maybe<Shape> FlatShape::ToShape() const {
  const auto& shape = std::make_shared<Shape>();
  JUST(ToShape(shape.get()));
  return shape;
}

Maybe<void> FlatShape::ToShape(Shape* shape) const {
  DimVector dim_vec;
  for (int i = 0; i < this->dim_size(); ++i) { dim_vec.emplace_back(this->dim(i)); }
  *shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
