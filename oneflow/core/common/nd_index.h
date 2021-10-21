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
#ifndef ONEFLOW_CORE_COMMON_ND_INDEX_H_
#define ONEFLOW_CORE_COMMON_ND_INDEX_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

class NdIndex final {
 public:
  NdIndex() = default;
  explicit NdIndex(const DimVector& dim_vec);
  NdIndex(const std::initializer_list<int64_t>& dim_vec);
  ~NdIndex() = default;
  NdIndex& operator=(const NdIndex& other);

  bool operator==(const NdIndex& rhs) const;
  bool operator!=(const NdIndex& rhs) const { return !(*this == rhs); }

  const DimVector& dim_vec() const { return dim_vec_; }

  int64_t At(int64_t index) const { return dim_vec_.at(index); }
  int64_t NumAxes() const { return dim_vec_.size(); }

 private:
  DimVector dim_vec_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ND_INDEX_H_
