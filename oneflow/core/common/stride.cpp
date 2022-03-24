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
#include "oneflow/core/common/stride.cfg.h"
#include "oneflow/core/common/stride_view.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

int64_t compute_index(int64_t out_offset, const StrideParam& in_stride,
                      const StrideParam& out_stride) {
  int64_t in_offset = 0;
  int64_t remaining = out_offset;
  for (size_t i = 0; i < in_stride.n_dim; ++i) {
    const int64_t idx = remaining / out_stride.stride[i];
    remaining -= idx * out_stride.stride[i];
    in_offset += idx * in_stride.stride[i];
  }
  return in_offset;
}

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
    int64_t stride = 1;
    for (size_t i = shape->NumAxes(); i > 0; --i) {
      stride_vec_.at(i - 1) = stride;
      stride *= shape->At(i - 1);
    }
  }
}

Stride::Stride(const std::initializer_list<int64_t>& stride_vec) : stride_vec_(stride_vec) {}
Stride::Stride(const StrideVector& stride_vec) : stride_vec_(stride_vec) {}
Stride::Stride(StrideVector&& stride_vec) : stride_vec_(std::move(stride_vec)) {}
Stride::Stride(const StrideProto& stride_proto) {
  stride_vec_.assign(stride_proto.dim().begin(), stride_proto.dim().end());
}
Stride::Stride(const cfg::StrideProto& stride_proto) {
  stride_vec_.assign(stride_proto.dim().begin(), stride_proto.dim().end());
}

Stride& Stride::assign(const StrideVector& stride_vec) {
  stride_vec_ = stride_vec;
  return *this;
}

Stride& Stride::CheckNumAxesIdenticalAndAssign(const StrideView& stride_view) {
  CHECK_EQ(NumAxes(), stride_view.NumAxes());
  std::copy(stride_view.ptr(), stride_view.ptr() + stride_view.NumAxes(), stride_vec_.data());
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

void Stride::ToProto(StrideProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(stride_vec_.begin(), stride_vec_.end());
}

}  // namespace oneflow
