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
  if (shape.is_initialized()) {
    const int64_t ndim = shape.NumAxes();
    stride_vec_.resize(shape.NumAxes());
    if (ndim > 0 && shape.elem_cnt() > 0) {
      std::exclusive_scan(shape.dim_vec().rbegin(), shape.dim_vec().rend(), stride_vec_.rbegin(), 1,
                          std::multiplies<>{});
    } else if (ndim > 0 && shape.elem_cnt() == 0) {
      // 0-size shape
      std::vector<int64_t> tmp_shape(ndim);
      for (int64_t i = 0; i < ndim; ++i) { tmp_shape[i] = shape.At(i) > 0 ? shape.At(i) : 1; }
      std::exclusive_scan(tmp_shape.rbegin(), tmp_shape.rend(), stride_vec_.rbegin(), 1,
                          std::multiplies<>{});
    }
  }
}

Stride::Stride(const std::shared_ptr<Shape>& shape) : Stride(*shape) {}

Stride::Stride(const std::initializer_list<int64_t>& stride_vec) : stride_vec_(stride_vec) {}
Stride::Stride(const DimVector& stride_vec) : stride_vec_(stride_vec) {}
Stride::Stride(DimVector&& stride_vec) : stride_vec_(std::move(stride_vec)) {}
Stride::Stride(const Int64ListProto& stride_proto) {
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

void Stride::ToProto(Int64ListProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(stride_vec_.begin(), stride_vec_.end());
}

}  // namespace oneflow
