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
#ifndef ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_IF_H_
#define ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_IF_H_

#include <string>
#include <vector>

#include "oneflow/core/common/symbol.h"

namespace oneflow {

class Device;

class NNGraphIf {
 public:
  virtual ~NNGraphIf() = default;

  virtual const std::string& job_name() const = 0;
  virtual const std::vector<std::string>& inputs_op_names() const = 0;
  virtual const std::vector<std::string>& outputs_op_names() const = 0;
  virtual const std::vector<bool>& inputs_valid() const = 0;
  virtual const std::vector<bool>& outputs_valid() const = 0;

 protected:
  NNGraphIf() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_IF_H_
