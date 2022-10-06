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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/optional.h"

namespace oneflow {

class ShapeView;
class MutShapeView;
class ShapeProto;

namespace cfg {
class ShapeProto;
}  // namespace cfg

/**
 * NOTE:
 *
 * There are two widely used shape-related classes: Shape and ShapeView.
 * The differences are:
 * 1. Shape owns the data, and ShapeView does not.
 * 2. ShapeView is very lightweight, whose size is only 16 bytes (two int64_t).
 *    So it should be passed by value.
 *
 * When adding new functions accepting a shape as a parameter, please follow
 * the rules:
 * 1. If your function doesn't modify the shape, prefer
 *    ShapeView. Shape can be implicitly converted to ShapeView so the method
 *    with ShapeView parameter can accept both Shape and ShapeView actually.
 * 2. If your function modify the shape but doesn't affect
 *    its rank, prefer MutShapeView. The reason is the same with rule 1.
 * 3. Use Shape otherwise.
 *
 * When adding new member methods of Shape or ShapeView, please follow
 * the rules:
 * 1. If the method is shared between Shape and ShapeView (like `NumAxes()`)
 *    please add it to ConstShapeMixIn.
 * 2. If the method is shared between Shape and MutShapeView (like `Set()`)
 *    please add it to MutShapeMixIn.
 * 3. Otherwise, add it to a concrete class (Shape, ShapeView or MutShapeView).
 *
 */
template<class T>
struct ConstShapeMixIn {
  using DimType = int64_t;

  int64_t NumAxes() const { return tp()->size(); }
  int64_t elem_cnt() const;
  int64_t At(int64_t index) const;
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t Count(int64_t begin_axis) const;
  bool Containing(ShapeView small_shape) const;
  bool MatchBeforeLastDim(ShapeView next_shape) const;
  std::string ToString() const;

  std::string DebugStr() const;

  void ToProto(ShapeProto* ret) const;

  template<typename StreamT>
  void SerializeWithTextFormat(StreamT& out_stream) const {
    for (int64_t dim : *this) { out_stream << std::to_string(dim) << ' '; }
  }

  bool operator==(const T& rhs) const;

 protected:
  // tp means "this pointer"
  T* tp() { return static_cast<T*>(this); }
  const T* tp() const { return static_cast<const T*>(this); }
};

template<class T>
struct MutShapeMixIn : public ConstShapeMixIn<T> {
  void Set(int64_t index, int64_t val) {
    CHECK_GE(index, 0);
    CHECK_LT(index, this->tp()->NumAxes())
        << " Shape: " << this->tp()->DebugStr() << " visit index: " << index
        << " > num_axes: " << this->tp()->NumAxes();
    (*this->tp())[index] = val;
  }
};

class Shape final : public DimVector, public MutShapeMixIn<Shape> {
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
  using DimVector::operator==;

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

  Shape& CheckNumAxesIdenticalAndAssign(ShapeView shape_view);
  Shape& LeftOnesExtendedAssign(ShapeView shape_view);

  // Getters and Setters
  bool is_initialized() const { return is_initialized_; }
  const DimVector& dim_vec() const { return *this; }
  DimVector& dim_vec() { return *this; }
  int64_t NumAxes() const {
    CHECK(is_initialized());
    return ConstShapeMixIn<Shape>::NumAxes();
  }
  AxisVector ShiftNegativeAxisVec(const AxisVector& axis_vec) const;
  Shape RemoveOnes(const AxisVector& axis_vec) const;
  static Shape Ones(const int64_t num_axes);
  AxisVector Axes4BroadcastTo(ShapeView broadcast_dim_vec) const;

  Maybe<Shape> Slice(int64_t start_dim, int64_t end_dim) const;

  bool operator==(const Shape& rhs) const;

 private:
  // Set default value here because some constructors are inherited from DimVector
  // TODO(daquexian): remove this field and make it initializied by construction
  bool is_initialized_ = true;
};

int64_t ShiftNegativeAxis(int64_t axis, const int64_t num_axes);

Shape CreateReducedShape(ShapeView shape, const AxisVector& axis_vec);
Shape CreateLeftExtendedShape(ShapeView shape, int ndims_extend_to);
Shape ExpandDimIf0D(const Shape& shape);
Shape ExpandDimIf0D(ShapeView shape);
Shape CreateReducedShapeOrOnesShape(ShapeView shape, const AxisVector& axis_vec);

std::ostream& operator<<(std::ostream& out, const Shape& shape);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::Shape> {
  size_t operator()(const oneflow::Shape& shape) const {
    if (!shape.is_initialized()) { return 0; }
    size_t ret = shape.NumAxes();
    FOR_RANGE(int, i, 0, shape.NumAxes()) { oneflow::AddHash(&ret, shape.At(i)); }
    return ret;
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_SHAPE_H_
