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
#ifndef ONEFLOW_ONEFLOW_API_PYTHON_RUN_INSTRUCTION_HELPER_H_
#define ONEFLOW_ONEFLOW_API_PYTHON_RUN_INSTRUCTION_HELPER_H_

#include "oneflow/core/common/global.h"
#include "oneflow/core/eager/eager_oneflow.h"

namespace oneflow {

std::string RunLogicalInstruction(
    const std::shared_ptr<vm::cfg::InstructionListProto>& cfg_instruction_list,
    const std::string& eager_symbol_list_str) {
  std::string error_str;
  Global<eager::EagerOneflow>::Get()
      ->RunLogicalInstruction(cfg_instruction_list, eager_symbol_list_str)
      .GetDataAndSerializedErrorProto(&error_str);
  // TODO(hanbinbin): return std::string is inefficient, and it will be solved when ErrorProto is
  // replaced by cfg::ErrorProto in the future.
  return error_str;
}

std::string RunPhysicalInstruction(
    const std::shared_ptr<vm::cfg::InstructionListProto>& cfg_instruction_list,
    const std::string& eager_symbol_list_str) {
  std::string error_str;
  Global<eager::EagerOneflow>::Get()
      ->RunPhysicalInstruction(cfg_instruction_list, eager_symbol_list_str)
      .GetDataAndSerializedErrorProto(&error_str);
  return error_str;
}

}  // namespace oneflow

#endif  // ONEFLOW_ONEFLOW_API_PYTHON_RUN_INSTRUCTION_HELPER_H_
