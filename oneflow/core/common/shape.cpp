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
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

template<class T>
int64_t ConstShapeMixIn<T>::elem_cnt() const {
  return std::accumulate(tp()->begin(), tp()->end(), int64_t(1), std::multiplies<>());
}

template<class T>
int64_t ConstShapeMixIn<T>::At(int64_t index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, tp()->NumAxes()) << " Shape: " << tp()->DebugStr() << " visit index: " << index
                                   << " > num_axes: " << tp()->NumAxes();
  return (*tp())[index];
}

template<class T>
int64_t ConstShapeMixIn<T>::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= tp()->NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= At(i); }
  return cnt;
}
template<class T>
int64_t ConstShapeMixIn<T>::Count(int64_t begin_axis) const {
  return Count(begin_axis, tp()->NumAxes());
}

template<class T>
bool ConstShapeMixIn<T>::Containing(ShapeView small_shape) const {
  if (tp()->NumAxes() < small_shape.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, small_shape.NumAxes()) {
    if (tp()->At(i) != small_shape.At(i)) { return false; }
  }
  return true;
}

template<class T>
bool ConstShapeMixIn<T>::MatchBeforeLastDim(ShapeView next_shape) const {
  if (tp()->NumAxes() != next_shape.NumAxes()) { return false; }
  for (int64_t i = 0; i < tp()->NumAxes() - 1; ++i) {
    if (next_shape.At(i) != tp()->At(i)) { return false; }
  }
  return true;
}

template<class T>
std::string ConstShapeMixIn<T>::ToString() const {
  std::stringstream ss;
  int32_t idx = 0;
  ss << "(";
  for (int64_t dim : *tp()) {
    ss << dim;
    if (++idx != tp()->size() || tp()->size() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

template<class T>
std::string ConstShapeMixIn<T>::DebugStr() const {
  return ToString();
}

template<class T>
void ConstShapeMixIn<T>::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(tp()->begin(), tp()->end());
}

template<class T>
bool ConstShapeMixIn<T>::operator==(const T& rhs) const {
  if (this->NumAxes() != rhs.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, this->NumAxes()) {
    if (this->At(i) != rhs.At(i)) { return false; }
  }
  return true;
}

template struct ConstShapeMixIn<Shape>;
template struct MutShapeMixIn<Shape>;
template struct ConstShapeMixIn<ShapeView>;
template struct ConstShapeMixIn<MutShapeView>;
template struct MutShapeMixIn<MutShapeView>;

Shape CreateReducedShape(ShapeView shape, const AxisVector& axis_vec) {
  // For 0-dim Tensor
  if (axis_vec.empty()) { return Shape({}); }
  DimVector dim_vec;
  shape.ToDimVector(&dim_vec);
  for (int64_t axis : axis_vec) { dim_vec.at(ShiftNegativeAxis(axis, shape.NumAxes())) = 1; }
  return Shape(std::move(dim_vec));
}

Shape CreateLeftExtendedShape(ShapeView shape, int ndims_left_extend_to) {
  CHECK_GE(ndims_left_extend_to, shape.NumAxes());
  DimVector dim_vec(ndims_left_extend_to);
  const size_t left_ones_num = ndims_left_extend_to - shape.NumAxes();
  int i = 0;
  for (; i < left_ones_num; ++i) { dim_vec.at(i) = 1LL; }
  for (; i < ndims_left_extend_to; ++i) { dim_vec.at(i) = shape.At(i - left_ones_num); }
  return Shape(std::move(dim_vec));
}

Shape ExpandDimIf0D(const Shape& shape) {
  if (shape.NumAxes() == 0) { return {1}; }
  return shape;
}

Shape ExpandDimIf0D(ShapeView shape) {
  if (shape.NumAxes() == 0) { return {1}; }
  return Shape(shape);
}

Shape CreateReducedShapeOrOnesShape(ShapeView shape, const AxisVector& axis_vec) {
  if (axis_vec.empty()) { return Shape::Ones(shape.NumAxes()); }
  return CreateReducedShape(shape, axis_vec);
}

int64_t ShiftNegativeAxis(int64_t axis, const int64_t num_axes) {
  if (axis < 0) { axis += num_axes; }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return axis;
}

Shape::Shape(const DimVector& dim_vec) : DimVector(dim_vec), is_initialized_(true) {}
Shape::Shape(DimVector&& dim_vec) : DimVector(std::move(dim_vec)), is_initialized_(true) {}
Shape::Shape(const ShapeProto& shape_proto)
    : DimVector(shape_proto.dim().begin(), shape_proto.dim().end()), is_initialized_(true) {}
Shape::Shape(ShapeView shape_view)
    : DimVector(shape_view.begin(), shape_view.end()), is_initialized_(true) {}

Shape& Shape::CheckNumAxesIdenticalAndAssign(ShapeView shape_view) {
  CHECK_EQ(NumAxes(), shape_view.NumAxes());
  std::copy(shape_view.ptr(), shape_view.ptr() + shape_view.NumAxes(), data());
  return *this;
}

Shape& Shape::LeftOnesExtendedAssign(ShapeView shape_view) {
  CHECK_GE(NumAxes(), shape_view.NumAxes());
  size_t left_ones_size = NumAxes() - shape_view.NumAxes();
  FOR_RANGE(int, i, 0, left_ones_size) { (*this)[i] = 1LL; }
  std::copy(shape_view.ptr(), shape_view.ptr() + shape_view.NumAxes(), data() + left_ones_size);
  return *this;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.DebugStr();
  return out;
}

AxisVector Shape::ShiftNegativeAxisVec(const AxisVector& axis_vec) const {
  const int64_t num_axes = this->NumAxes();
  AxisVector ret = axis_vec;
  for (int64_t i = 0; i < axis_vec.size(); i++) {
    ret.at(i) = ShiftNegativeAxis(axis_vec.at(i), num_axes);
  }
  return ret;
}

Shape Shape::RemoveOnes(const AxisVector& axis_vec) const {
  DimVector dim_vec;
  const AxisVector& axis_vec_shifted = ShiftNegativeAxisVec(axis_vec);
  for (int64_t i = 0; i < this->dim_vec().size(); i++) {
    if (std::find(axis_vec_shifted.begin(), axis_vec_shifted.end(), i) == axis_vec_shifted.end()) {
      dim_vec.emplace_back(this->dim_vec().at(i));
    } else {
      CHECK_EQ(this->dim_vec().at(i), 1);
    }
  }
  return Shape(dim_vec);
}

Shape Shape::Ones(const int64_t num_axes) {
  DimVector dim_vec(num_axes);
  std::fill(dim_vec.begin(), dim_vec.end(), 1);
  return Shape(dim_vec);
}

AxisVector Shape::Axes4BroadcastTo(ShapeView broadcast_shape) const {
  AxisVector broadcast_axis_vec;
  CHECK_EQ(broadcast_shape.NumAxes(), NumAxes());
  for (int64_t i = 0; i < NumAxes(); i++) {
    if (this->dim_vec().at(i) != broadcast_shape[i] && this->dim_vec().at(i) == 1) {
      broadcast_axis_vec.emplace_back(i);
    } else {
      CHECK_EQ(this->dim_vec().at(i), broadcast_shape[i]);
    }
  }
  CHECK(!broadcast_axis_vec.empty());
  return broadcast_axis_vec;
}

Maybe<Shape> Shape::Slice(int64_t start_dim, int64_t end_dim) const {
  CHECK_OR_RETURN(start_dim >= 0 && end_dim >= start_dim);
  int64_t ndims = this->NumAxes();
  if (start_dim > ndims) { start_dim = ndims; }
  if (end_dim > ndims) { end_dim = ndims; }
  std::shared_ptr<Shape> shape = std::make_shared<Shape>();
  for (int64_t i = start_dim; i < end_dim && i < ndims; ++i) { shape->emplace_back(this->At(i)); }
  return shape;
}

bool Shape::operator==(const Shape& rhs) const {
  if (is_initialized_ != rhs.is_initialized_) { return false; }
  if (is_initialized_ == false) { return true; }
  if (this->NumAxes() != rhs.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, this->NumAxes()) {
    if (this->At(i) != rhs.At(i)) { return false; }
  }
  return true;
}

}  // namespace oneflow
