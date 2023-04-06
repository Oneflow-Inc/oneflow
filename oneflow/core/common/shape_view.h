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
#ifndef ONEFLOW_CORE_REGISTER_SHAPE_VIEW_H_
#define ONEFLOW_CORE_REGISTER_SHAPE_VIEW_H_

#include "oneflow/core/common/array_ref.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

class ShapeProto;
class Shape;

class ShapeView : public ArrayRef<Dim>, public ConstShapeMixIn<ShapeView> {
 public:
  ShapeView() = default;
  // NOLINTNEXTLINE
  ShapeView(const Shape& shape) : ArrayRef<Dim>(shape.dim_vec().data(), shape.dim_vec().size()){};
  ShapeView(const int64_t* start, size_t size)
      : ArrayRef<Dim>(reinterpret_cast<const Dim*>(start), size) {}

  using Base = ArrayRef<Dim>;
  using ArrayRef<Dim>::ArrayRef;

  // NOTE(daquexian): At(int64_t index) returns int64_t instead of Dim for better
  // compatibility with old code. Please use DimAt(int64_t index) if the element
  // may not be an integer.
  const int64_t& At(int64_t index) const;
  // NOTE(daquexian): operator[](int64_t index) returns int64_t instead of Dim for better
  // compatibility with old code. Please use DimAt(int64_t index) if the element
  // may not be an integer.
  const int64_t& operator[](int64_t index) const { return At(index); }
  // NOTE(daquexian): data() returns int64_t* instead of Dim* for better
  // compatibility with old code. It is recommended to use int64_ptr()
  // whenever possible.
  const int64_t* data() const {
    CHECK(all_dims_known());
    return reinterpret_cast<const int64_t*>(ArrayRef::data());
  }
  // NOTE(daquexian): ptr() returns int64_t* instead of Dim* for better
  // compatibility with old code. It is recommended to use int64_ptr()
  // whenever possible.
  const int64_t* ptr() const { return data(); }

  void ToDimVector(DimVector* dim_vec) const;
  void ToShape(Shape* shape) const;
};

std::ostream& operator<<(std::ostream& out, ShapeView shape);

class MutShapeView final : public MutableArrayRef<Dim>, public MutShapeMixIn<MutShapeView> {
 public:
  using Base = MutableArrayRef<Dim>;
  using MutableArrayRef<Dim>::MutableArrayRef;
  // NOLINTNEXTLINE
  MutShapeView(Shape& shape)
      : MutableArrayRef<Dim>(shape.dim_vec().data(), shape.dim_vec().size()){};
  MutShapeView(int64_t* start, size_t size)
      : MutableArrayRef<Dim>(reinterpret_cast<Dim*>(start), size) {}

  const Dim& DimAt(int64_t index) const;
  Dim& DimAt(int64_t index);
  // NOTE(daquexian): At(int64_t index) returns int64_t instead of Dim for better
  // compatibility with old code. Please use DimAt(int64_t index) if the element
  // may not be an integer.
  const int64_t& At(int64_t index) const;
  // NOTE(daquexian): operator[](int64_t index) returns int64_t instead of Dim for better
  // compatibility with old code. Please use DimAt(int64_t index) if the element
  // may not be an integer.
  const int64_t& operator[](int64_t index) const { return At(index); }
  // NOTE(daquexian): At(int64_t index) returns int64_t instead of Dim for better
  // compatibility with old code. Please use DimAt(int64_t index) if the element
  // may not be an integer.
  int64_t& At(int64_t index);
  // NOTE(daquexian): operator[](int64_t index) returns int64_t instead of Dim for better
  // compatibility with old code. Please use DimAt(int64_t index) if the element
  // may not be an integer.
  int64_t& operator[](int64_t index) { return At(index); }
  // NOTE(daquexian): data() returns int64_t* instead of Dim* for better
  // compatibility with old code. It is recommended to use int64_ptr()
  // whenever possible.
  const int64_t* data() const {
    CHECK(all_dims_known());
    return reinterpret_cast<const int64_t*>(MutableArrayRef::data());
  }
  // NOTE(daquexian): data() returns int64_t* instead of Dim* for better
  // compatibility with old code. It is recommended to use int64_ptr()
  // whenever possible.
  int64_t* data() {
    for (Dim dim : *this) { CHECK(dim.is_known()); }
    return reinterpret_cast<int64_t*>(MutableArrayRef::data());
  }

  void set_shape(ShapeView shape);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_SHAPE_VIEW_H_
