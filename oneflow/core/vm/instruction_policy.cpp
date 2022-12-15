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
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void InstructionPolicy::InitInstructionStatus(Instruction* instruction) {
  instruction->stream_policy().InitInstructionStatus(instruction->stream(),
                                                     instruction->mut_status_buffer());
}

void InstructionPolicy::DeleteInstructionStatus(Instruction* instruction) {
  instruction->stream_policy().DeleteInstructionStatus(instruction->stream(),
                                                       instruction->mut_status_buffer());
}

}  // namespace vm
}  // namespace oneflow
