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
    resize(shape.NumAxes());
    if (ndim > 0 && shape.elem_cnt() > 0) {
      std::exclusive_scan(shape.dim_vec().rbegin(), shape.dim_vec().rend(), rbegin(), (int64_t)1,
                          std::multiplies<>{});
    } else if (ndim > 0 && shape.elem_cnt() == 0) {
      // 0-size shape
      std::vector<int64_t> tmp_shape(ndim);
      for (int64_t i = 0; i < ndim; ++i) { tmp_shape[i] = shape.At(i) > 0 ? shape.At(i) : 1; }
      std::exclusive_scan(tmp_shape.rbegin(), tmp_shape.rend(), rbegin(), (int64_t)1,
                          std::multiplies<>{});
    }
  }
}

Stride::Stride(const std::shared_ptr<Shape>& shape) : Stride(*shape) {}

Stride::Stride(const Int64ListProto& stride_proto)
    : DimVector(stride_proto.dim().begin(), stride_proto.dim().end()) {}

Stride& Stride::CheckNumAxesIdenticalAndAssign(const Stride& stride) {
  CHECK_EQ(size(), stride.size());
  assign(stride);
  return *this;
}

std::string Stride::ToString() const {
  std::stringstream ss;
  int32_t idx = 0;
  ss << "(";
  for (int64_t dim : *this) {
    ss << dim;
    if (++idx != this->size() || this->size() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

void Stride::ToProto(Int64ListProto* ret) const {
  *(ret->mutable_dim()) = PbRf<int64_t>(begin(), end());
}

std::ostream& operator<<(std::ostream& out, const Stride& stride) {
  out << stride.ToString();
  return out;
}

}  // namespace oneflow
