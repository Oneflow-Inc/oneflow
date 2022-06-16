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

class ShapeView : public ArrayRef<int64_t>, public ConstShapeMixIn<ShapeView> {
 public:
  ShapeView() = default;
  // NOLINTNEXTLINE
  ShapeView(const ShapeProto& shape_proto)
      : ArrayRef<int64_t>(shape_proto.dim().data(), shape_proto.dim_size()){};
  // NOLINTNEXTLINE
  ShapeView(const Shape& shape)
      : ArrayRef<int64_t>(shape.dim_vec().data(), shape.dim_vec().size()){};

  using ArrayRef<DimType>::ArrayRef;

  const DimType* ptr() const { return this->data(); }

  void ToDimVector(DimVector* dim_vec) const;
  void ToShape(Shape* shape) const;
};

std::ostream& operator<<(std::ostream& out, ShapeView shape);

class MutShapeView final : public MutableArrayRef<int64_t>, public MutShapeMixIn<MutShapeView> {
 public:
  using MutableArrayRef<DimType>::MutableArrayRef;
  // NOLINTNEXTLINE
  MutShapeView(Shape& shape)
      : MutableArrayRef<int64_t>(shape.dim_vec().data(), shape.dim_vec().size()){};

  int64_t* mut_ptr() const { return this->data(); }

  void set_shape(ShapeView shape);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_SHAPE_VIEW_H_
