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
#include "oneflow/core/common/shape.pb.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

ShapeView::ShapeView(const ShapeProto& shape_proto)
    : ShapeViewBase<const int64_t>(shape_proto.dim().data(), shape_proto.dim_size()) {}
ShapeView::ShapeView(const Shape& shape)
    : ShapeViewBase<const int64_t>(shape.dim_vec().data(), shape.dim_vec().size()) {}

template<typename DimT>
int64_t ShapeViewBase<DimT>::At(int64_t index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_axes_);
  return ptr_[index];
}

template<typename DimT>
int64_t ShapeViewBase<DimT>::Count(int64_t begin_axis) const {
  return this->Count(begin_axis, NumAxes());
}

template<typename DimT>
int64_t ShapeViewBase<DimT>::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= this->NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= this->At(i); }
  return cnt;
}

template<typename DimT>
int64_t ShapeViewBase<DimT>::elem_cnt() const {
  return this->Count(0);
}

template<typename DimT>
std::string ShapeViewBase<DimT>::ToString() const {
  std::stringstream ss;
  ss << "(";
  FOR_RANGE(int, i, 0, this->NumAxes()) {
    int64_t dim = this->At(i);
    ss << dim;
    if (i != this->NumAxes() - 1 || this->NumAxes() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

template<typename DimT>
void ShapeViewBase<DimT>::ToDimVector(DimVector* dim_vec) const {
  dim_vec->resize(num_axes_);
  dim_vec->assign(ptr_, ptr_ + num_axes_);
}

template<typename DimT>
void ShapeViewBase<DimT>::ToShape(Shape* shape) const {
  DimVector dim_vec;
  this->ToDimVector(&dim_vec);
  *shape = Shape(std::move(dim_vec));
}

template class ShapeViewBase<const int64_t>;
template class ShapeViewBase<int64_t>;

std::ostream& operator<<(std::ostream& out, const ShapeView& shape) {
  out << shape.ToString();
  return out;
}

void MutShapeView::Set(int64_t axis, int64_t val) {
  CHECK_GE(axis, 0);
  CHECK_LT(axis, NumAxes());
  dim_ptr()[axis] = val;
}

void MutShapeView::set_shape(const Shape& shape) {
  CHECK_EQ(NumAxes(), shape.NumAxes());
  std::copy(shape.dim_vec().data(), shape.dim_vec().data() + shape.NumAxes(), dim_ptr());
}

void MutShapeView::set_shape(const ShapeView& shape) {
  CHECK_EQ(NumAxes(), shape.NumAxes());
  std::copy(shape.ptr(), shape.ptr() + shape.NumAxes(), dim_ptr());
}

}  // namespace oneflow
