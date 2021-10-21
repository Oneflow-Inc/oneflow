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
#include "oneflow/core/common/nd_index.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

NdIndex::NdIndex(const std::initializer_list<int64_t>& dim_vec) : dim_vec_(dim_vec) {}

NdIndex::NdIndex(const DimVector& dim_vec) : dim_vec_(dim_vec) {}

NdIndex& NdIndex::operator=(const NdIndex& shape) {
  dim_vec_ = shape.dim_vec_;
  return *this;
}

bool NdIndex::operator==(const NdIndex& rhs) const { return dim_vec_ == rhs.dim_vec_; }

}  // namespace oneflow
