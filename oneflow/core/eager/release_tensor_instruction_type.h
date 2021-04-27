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
#ifndef ONEFLOW_CORE_EAGER_RELEASE_TENSOR_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_EAGER_RELEASE_TENSOR_INSTRUCTION_TYPE_H_

#include "oneflow/core/vm/instruction_type.h"

namespace oneflow {

namespace vm {

class ReleaseTensorInstructionType : public vm::InstructionType {
 public:
  ReleaseTensorInstructionType() = default;
  ~ReleaseTensorInstructionType() override = default;

  void Infer(vm::Instruction* instruction) const override;
  void Compute(vm::Instruction* instruction) const override;
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_RELEASE_TENSOR_INSTRUCTION_TYPE_H_
