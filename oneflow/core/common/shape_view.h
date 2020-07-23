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

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow {

class ShapeProto;
class Shape;

template<typename DimT>
class ShapeViewBase {
 public:
  using DimType = DimT;
  ShapeViewBase(DimType* ptr, int64_t num_axes) : ptr_(ptr), num_axes_(num_axes) {}
  ShapeViewBase(const ShapeViewBase& rhs) = default;
  ~ShapeViewBase() = default;

  int64_t NumAxes() const { return num_axes_; }
  int64_t At(int64_t index) const;
  int64_t Count(int64_t begin_axis) const;
  int64_t Count(int64_t begin_axis, int64_t end_axis) const;
  int64_t elem_cnt() const;
  const DimType* ptr() const { return ptr_; }

  bool operator==(const ShapeViewBase& rhs) const;
  std::string ToString() const;
  void ToDimVector(DimVector* dim_vec) const;
  void ToShape(Shape* shape) const;

  void set_ptr(DimType* ptr) { ptr_ = ptr; }

 protected:
  DimType* dim_ptr() const { return ptr_; }

 private:
  DimType* ptr_;
  int64_t num_axes_;
};

class ShapeView final : public ShapeViewBase<const int64_t> {
 public:
  ShapeView() : ShapeViewBase<const int64_t>(nullptr, 0) {}
  ShapeView(const int64_t* ptr, int64_t num_axes) : ShapeViewBase<const int64_t>(ptr, num_axes) {}
  ShapeView(const ShapeProto& shape_proto);
  ShapeView(const Shape& shape);
  ShapeView(const ShapeView& rhs) = default;
  ~ShapeView() = default;
};

std::ostream& operator<<(std::ostream& out, const ShapeView& shape);

class MutShapeView final : public ShapeViewBase<int64_t> {
 public:
  MutShapeView() : ShapeViewBase<int64_t>(nullptr, 0) {}
  MutShapeView(int64_t* ptr, int64_t num_axes) : ShapeViewBase<int64_t>(ptr, num_axes) {}
  MutShapeView(const MutShapeView& rhs) = default;
  ~MutShapeView() = default;

  int64_t* mut_ptr() const { return dim_ptr(); }
  void Set(int64_t axis, int64_t val);

  void set_shape(const Shape& val);
  void set_shape(const ShapeView& shape);
};

template<typename DimT>
bool ShapeViewBase<DimT>::operator==(const ShapeViewBase<DimT>& rhs) const {
  if (this->NumAxes() != rhs.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, this->NumAxes()) {
    if (At(i) != rhs.At(i)) { return false; }
  }
  return true;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_SHAPE_VIEW_H_
