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
#ifndef ONEFLOW_USER_DATA_PARSER_H_
#define ONEFLOW_USER_DATA_PARSER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
class Parser {
 public:
  using SampleType = LoadTarget;
  using BatchType = std::vector<SampleType>;

  Parser() = default;
  virtual ~Parser() = default;

  virtual void Parse(BatchType& batch_data, user_op::KernelComputeContext* ctx) = 0;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_PARSER_H_
