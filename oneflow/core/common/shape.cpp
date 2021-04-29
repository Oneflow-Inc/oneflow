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
#include "oneflow/core/common/shape.cfg.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

Shape CreateReducedShape(const ShapeView& shape, const AxisVector& axis_vec) {
  CHECK_EQ(axis_vec.empty(), false);
  DimVector dim_vec;
  shape.ToDimVector(&dim_vec);
  for (int64_t axis : axis_vec) { dim_vec.at(ShiftNegativeAxis(axis, shape.NumAxes())) = 1; }
  return Shape(std::move(dim_vec));
}

Shape CreateLeftExtendedShape(const ShapeView& shape, int ndims_left_extend_to) {
  CHECK_GE(ndims_left_extend_to, shape.NumAxes());
  DimVector dim_vec(ndims_left_extend_to);
  const size_t left_ones_num = ndims_left_extend_to - shape.NumAxes();
  int i = 0;
  for (; i < left_ones_num; ++i) { dim_vec.at(i) = 1LL; }
  for (; i < ndims_left_extend_to; ++i) { dim_vec.at(i) = shape.At(i - left_ones_num); }
  return Shape(std::move(dim_vec));
}

Shape CreateReducedShapeOrOnesShape(const ShapeView& shape, const AxisVector& axis_vec) {
  if (axis_vec.empty()) { return Shape::Ones(shape.NumAxes()); }
  return CreateReducedShape(shape, axis_vec);
}

int64_t ShiftNegativeAxis(int64_t axis, const int64_t num_axes) {
  if (axis < 0) { axis += num_axes; }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return axis;
}

Shape::Shape(const std::initializer_list<int64_t>& dim_vec) : dim_vec_(dim_vec) { UpdateElemCnt(); }
Shape::Shape(const DimVector& dim_vec) : dim_vec_(dim_vec) { UpdateElemCnt(); }
Shape::Shape(DimVector&& dim_vec) : dim_vec_(std::move(dim_vec)) { UpdateElemCnt(); }

Shape::Shape(const ShapeProto& shape_proto) {
  dim_vec_.assign(shape_proto.dim().begin(), shape_proto.dim().end());
  UpdateElemCnt();
}

Shape::Shape(const cfg::ShapeProto& shape_proto) {
  dim_vec_.assign(shape_proto.dim().begin(), shape_proto.dim().end());
  UpdateElemCnt();
}

Shape& Shape::operator=(const Shape& shape) {
  dim_vec_ = shape.dim_vec_;
  UpdateElemCnt();
  return *this;
}

Shape& Shape::CheckNumAxesIdenticalAndAssign(const ShapeView& shape_view) {
  CHECK_EQ(NumAxes(), shape_view.NumAxes());
  std::copy(shape_view.ptr(), shape_view.ptr() + shape_view.NumAxes(), dim_vec_.data());
  UpdateElemCnt();
  return *this;
}

Shape& Shape::LeftOnesExtendedAssign(const ShapeView& shape_view) {
  CHECK_GE(NumAxes(), shape_view.NumAxes());
  size_t left_ones_size = NumAxes() - shape_view.NumAxes();
  FOR_RANGE(int, i, 0, left_ones_size) { dim_vec_.at(i) = 1LL; }
  std::copy(shape_view.ptr(), shape_view.ptr() + shape_view.NumAxes(),
            dim_vec_.data() + left_ones_size);
  UpdateElemCnt();
  return *this;
}

bool Shape::operator==(const Shape& rhs) const { return dim_vec_ == rhs.dim_vec_; }

std::string Shape::ToString() const {
  std::stringstream ss;
  int32_t idx = 0;
  ss << "(";
  for (int64_t dim : dim_vec_) {
    ss << dim;
    if (++idx != dim_vec_.size() || dim_vec_.size() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

std::string Shape::DebugStr() const { return ToString(); }

void Shape::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(dim_vec_.begin(), dim_vec_.end());
}

void Shape::Set(int64_t index, int64_t val) {
  dim_vec_.at(index) = val;
  UpdateElemCnt();
}

int64_t Shape::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= At(i); }
  return cnt;
}

int64_t Shape::Count(int64_t begin_axis) const { return Count(begin_axis, NumAxes()); }

void Shape::UpdateElemCnt() {
  elem_cnt_ = 1;
  for (int64_t s : dim_vec_) { elem_cnt_ *= s; }
  if (dim_vec_.size() == 0) { elem_cnt_ = 0; }
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
      dim_vec.push_back(this->dim_vec().at(i));
    } else {
      CHECK_EQ(this->dim_vec().at(i), 1);
    }
  }
  if (dim_vec.empty()) { dim_vec.push_back(1); }
  return Shape(dim_vec);
}

Shape Shape::Ones(const int64_t num_axes) {
  DimVector dim_vec(num_axes);
  std::fill(dim_vec.begin(), dim_vec.end(), 1);
  return Shape(dim_vec);
}

AxisVector Shape::Axes4BroadcastTo(const Shape& broadcast_shape) const {
  AxisVector broadcast_axis_vec;
  CHECK_EQ(broadcast_shape.NumAxes(), NumAxes());
  for (int64_t i = 0; i < NumAxes(); i++) {
    if (this->dim_vec().at(i) != broadcast_shape.dim_vec().at(i) && this->dim_vec().at(i) == 1) {
      broadcast_axis_vec.push_back(i);
    } else {
      CHECK_EQ(this->dim_vec().at(i), broadcast_shape.dim_vec().at(i));
    }
  }
  CHECK(!broadcast_axis_vec.empty());
  return broadcast_axis_vec;
}

bool Shape::Containing(const Shape& small_shape) const {
  if (this->NumAxes() < small_shape.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, small_shape.NumAxes()) {
    if (this->At(i) != small_shape.At(i)) { return false; }
  }
  return true;
}

}  // namespace oneflow
