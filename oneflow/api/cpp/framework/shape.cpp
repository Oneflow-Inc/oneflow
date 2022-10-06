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
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow_api {

namespace of = oneflow;
namespace {

of::DimVector ToOneflowDimVcetor(const std::vector<int64_t>& dim_vec) {
  return of::DimVector(dim_vec.begin(), dim_vec.end());
}

}  // namespace

Shape::Shape() : shape_(std::make_shared<of::Shape>(of::Shape({0}))) {}

Shape::Shape(const std::vector<int64_t>& dim_vec)
    : shape_(std::make_shared<of::Shape>(ToOneflowDimVcetor(dim_vec))) {}

Shape::Shape(const std::initializer_list<int64_t>& dim_vec)
    : shape_(std::make_shared<of::Shape>(dim_vec)) {}

Shape& Shape::operator=(const Shape& shape) {
  this->shape_.reset();
  this->shape_ = shape.shape_;
  return *this;
}

bool Shape::operator==(const Shape& rhs) const { return *shape_ == *rhs.shape_; }

bool Shape::operator!=(const Shape& rhs) const { return !(*this == rhs); }

int64_t Shape::elem_cnt() const { return shape_->elem_cnt(); }

int64_t Shape::At(int64_t index) const { return shape_->At(index); }

void Shape::Set(int64_t index, int64_t val) { shape_->Set(index, val); }

int64_t Shape::NumAxes() const { return shape_->NumAxes(); }

int64_t Shape::Count(int64_t begin_axis, int64_t end_axis) const {
  return shape_->Count(begin_axis, end_axis);
}

int64_t Shape::Count(int64_t begin_axis) const { return shape_->Count(begin_axis); }

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  os << shape.shape_->DebugStr();
  return os;
}

}  // namespace oneflow_api
