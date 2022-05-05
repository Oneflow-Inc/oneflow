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

#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/cplusplus_17.h"

namespace oneflow {

Stride::Stride(const Shape& shape) {
  if (shape.NumAxes() > 0) {
    stride_vec_.resize(shape.NumAxes());
    int64_t stride = 1;
    for (size_t i = shape.NumAxes(); i > 0; --i) {
      stride_vec_.at(i - 1) = stride;
      stride *= shape.At(i - 1);
    }
  }
}

Stride::Stride(const std::shared_ptr<Shape>& shape) {
  if (shape->NumAxes() > 0) {
    stride_vec_.resize(shape->NumAxes());
    std::exclusive_scan(shape->dim_vec().rbegin(), shape->dim_vec().rend(), stride_vec_.begin(), 1,
                        std::multiplies<>{});
  }
}

Stride::Stride(const std::initializer_list<int64_t>& stride_vec) : stride_vec_(stride_vec) {}
Stride::Stride(const DimVector& stride_vec) : stride_vec_(stride_vec) {}
Stride::Stride(DimVector&& stride_vec) : stride_vec_(std::move(stride_vec)) {}
Stride::Stride(const ShapeProto& stride_proto) {
  stride_vec_.assign(stride_proto.dim().begin(), stride_proto.dim().end());
}
Stride::Stride(const cfg::ShapeProto& stride_proto) {
  stride_vec_.assign(stride_proto.dim().begin(), stride_proto.dim().end());
}

Stride& Stride::assign(const DimVector& stride_vec) {
  stride_vec_ = stride_vec;
  return *this;
}

Stride& Stride::CheckNumAxesIdenticalAndAssign(const Stride& stride) {
  CHECK_EQ(NumAxes(), stride.NumAxes());
  stride_vec_.assign(stride.StrideVec().begin(), stride.StrideVec().end());
  return *this;
}

Stride& Stride::operator=(const Stride& stride) {
  stride_vec_ = stride.stride_vec_;
  return *this;
}

bool Stride::operator==(const Stride& rhs) const { return stride_vec_ == rhs.stride_vec_; }

std::string Stride::ToString() const {
  std::stringstream ss;
  int32_t idx = 0;
  ss << "(";
  for (int64_t dim : stride_vec_) {
    ss << dim;
    if (++idx != stride_vec_.size() || stride_vec_.size() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

void Stride::ToProto(ShapeProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(stride_vec_.begin(), stride_vec_.end());
}

}  // namespace oneflow
