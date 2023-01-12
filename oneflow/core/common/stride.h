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

#ifndef ONEFLOW_CORE_FRAMEWORK_STRIDE_H_
#define ONEFLOW_CORE_FRAMEWORK_STRIDE_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/sequential.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class Int64ListProto;

class Stride final : public DimVector {
 public:
  Stride() = default;
  using DimVector::DimVector;
  explicit Stride(const Shape& shape);
  explicit Stride(const std::shared_ptr<Shape>& shape);
  explicit Stride(const Int64ListProto& stride_proto);
  Stride& CheckNumAxesIdenticalAndAssign(const Stride& stride);
  ~Stride() = default;

  std::string ToString() const;
  void ToProto(Int64ListProto*) const;
};

std::ostream& operator<<(std::ostream& out, const Stride& stride);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::Stride> {
  size_t operator()(const oneflow::Stride& stride) const {
    size_t ret = stride.size();
    FOR_RANGE(int, i, 0, stride.size()) { oneflow::AddHash(&ret, stride.at(i)); }
    return ret;
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_STRIDE_H_
