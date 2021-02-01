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
#include "oneflow/core/framework/op_interpreter.h"

namespace oneflow {
namespace one {

std::vector<TensorRef> LazyOpInterpreter::Interpret(const Operation& op,
                                                    const std::vector<TensorRef>& inputs) {
  return std::vector<TensorRef>{};
}

std::vector<TensorRef> EagerOpInterpreter::Interpret(const Operation& op,
                                                     const std::vector<TensorRef>& inputs) {
  return std::vector<TensorRef>{};
}

}  // namespace one
}  // namespace oneflow
