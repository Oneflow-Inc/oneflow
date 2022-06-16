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
#ifndef ONEFLOW_CORE_COMMON_SHAPE_H_
#define ONEFLOW_CORE_COMMON_SHAPE_H_

#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/optional.h"

namespace oneflow {

class ShapeView;
class ShapeProto;

namespace cfg {
class ShapeProto;
}  // namespace cfg

class Shape final : public DimVector {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Shape);
  using DimVector::DimVector;
  Shape() : is_initialized_(false) {}
  explicit Shape(const DimVector& dim_vec);
  explicit Shape(DimVector&& dim_vec);
  explicit Shape(const ShapeProto& shape_proto);
  // explicit constructor from ShapeView
  explicit Shape(ShapeView shape_view);
  ~Shape() = default;

#define OVERRIDE_ADD_DATA_FUNC(func)              \
  template<typename... Args>                      \
  void func(Args... args) {                       \
    DimVector::func(std::forward<Args>(args)...); \
    is_initialized_ = true;                       \
  }

  OVERRIDE_ADD_DATA_FUNC(assign)
  OVERRIDE_ADD_DATA_FUNC(push_back)
  OVERRIDE_ADD_DATA_FUNC(emplace_back)
  OVERRIDE_ADD_DATA_FUNC(append)
  OVERRIDE_ADD_DATA_FUNC(insert)
  OVERRIDE_ADD_DATA_FUNC(resize)

#undef OVERRIDE_ADD_DATA_FUNC

  Shape& CheckNumAxesIdenticalAndAssign(const ShapeView& shape_view);
  Shape& LeftOnesExtendedAssign(const ShapeView& shape_view);

  std::string DebugStr() const;
  std::string ToString() const;

  void ToProto(ShapeProto*) const;

  template<typename StreamT>
  void SerializeWithTextFormat(StreamT& out_stream) const;

  // Getters and Setters
  bool is_initialized() const { return is_initialized_; }
  const DimVector& dim_vec() const { return *this; }
  DimVector& dim_vec() { return *this; }
  int64_t elem_cnt() const {
    return std::accumulate(begin(), end(), int64_t(1), std::multiplies<>());
  }
  int64_t At(int64_t index) const;
  void Set(int64_t index, int64_t val);
  int64_t NumAxes() const {
    CHECK(is_initialized());
    return size();
  }
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t Count(int64_t begin_axis) const;

  AxisVector ShiftNegativeAxisVec(const AxisVector& axis_vec) const;
  Shape RemoveOnes(const AxisVector& axis_vec) const;
  static Shape Ones(const int64_t num_axes);
  AxisVector Axes4BroadcastTo(const Shape& broadcast_dim_vec) const;

  bool Containing(const Shape& small_shape) const;
  bool MatchBeforeLastDim(const Shape& next_shape) const;

  Maybe<Shape> Slice(int64_t start_dim, int64_t end_dim) const;

  ShapeView ToShapeView() const { return ShapeView(data(), size()); }

  MutShapeView ToMutShapeView() { return MutShapeView(data(), size()); }

 private:
  // Set default value here because some constructors are inherited from DimVector
  // TODO(daquexian): remove this field and make it initializied by construction
  bool is_initialized_ = true;
};

int64_t ShiftNegativeAxis(int64_t axis, const int64_t num_axes);

Shape CreateReducedShape(const ShapeView& shape, const AxisVector& axis_vec);
Shape CreateLeftExtendedShape(const ShapeView& shape, int ndims_extend_to);
Shape ZeroDimCompatiableShape(const Shape& shape);
Shape CreateReducedShapeOrOnesShape(const ShapeView& shape, const AxisVector& axis_vec);
template<typename StreamT>
void Shape::SerializeWithTextFormat(StreamT& out_stream) const {
  for (int64_t dim : *this) { out_stream << std::to_string(dim) << ' '; }
}

std::ostream& operator<<(std::ostream& out, const Shape& shape);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::Shape> {
  size_t operator()(const oneflow::Shape& shape) const {
    size_t ret = shape.NumAxes();
    FOR_RANGE(int, i, 0, shape.NumAxes()) { oneflow::AddHash(&ret, shape.At(i)); }
    return ret;
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_SHAPE_H_
