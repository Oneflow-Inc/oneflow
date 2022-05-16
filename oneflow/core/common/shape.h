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

class Shape final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(Shape);
  Shape() : is_initialized_(false) {}
  explicit Shape(const DimVector& dim_vec);
  explicit Shape(DimVector&& dim_vec);
  explicit Shape(const ShapeProto& shape_proto);
  Shape(const std::initializer_list<int64_t>& dim_vec);
  ~Shape() = default;
  Shape& operator=(const Shape& shape);
  Shape& assign(const DimVector& dim_vec);
  Shape& CheckNumAxesIdenticalAndAssign(const ShapeView& shape_view);
  Shape& LeftOnesExtendedAssign(const ShapeView& shape_view);

  bool operator==(const Shape& rhs) const;
  bool operator!=(const Shape& rhs) const { return !(*this == rhs); }
  std::string DebugStr() const;
  std::string ToString() const;

  void ToProto(ShapeProto*) const;

  template<typename StreamT>
  void SerializeWithTextFormat(StreamT& out_stream) const;

  // Getters and Setters
  bool is_initialized() const { return is_initialized_; }
  const DimVector& dim_vec() const { return dim_vec_; }
  DimVector& dim_vec() { return dim_vec_; }
  int64_t elem_cnt() const {
    return std::accumulate(dim_vec_.begin(), dim_vec_.end(), int64_t(1), std::multiplies<>());
  }
  int64_t At(int64_t index) const;
  void Set(int64_t index, int64_t val);
  int64_t NumAxes() const {
    CHECK(is_initialized());
    return dim_vec_.size();
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

  ShapeView ToShapeView() const { return ShapeView(dim_vec_.data(), dim_vec_.size()); }

  MutShapeView ToMutShapeView() { return MutShapeView(dim_vec_.data(), dim_vec_.size()); }

 private:
  DimVector dim_vec_;
  bool is_initialized_;
};

int64_t ShiftNegativeAxis(int64_t axis, const int64_t num_axes);

Shape CreateReducedShape(const ShapeView& shape, const AxisVector& axis_vec);
Shape CreateLeftExtendedShape(const ShapeView& shape, int ndims_extend_to);
Shape ZeroDimCompatiableShape(const Shape& shape);
Shape CreateReducedShapeOrOnesShape(const ShapeView& shape, const AxisVector& axis_vec);
template<typename StreamT>
void Shape::SerializeWithTextFormat(StreamT& out_stream) const {
  for (int64_t dim : dim_vec_) { out_stream << std::to_string(dim) << ' '; }
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
