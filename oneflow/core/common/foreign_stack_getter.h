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
#ifndef ONEFLOW_CORE_COMMON_FOREIGN_STACK_GETTER_H
#define ONEFLOW_CORE_COMMON_FOREIGN_STACK_GETTER_H

#include <cstdint>

inline int* GetInstructionIdThisThreadInternal() {
  static thread_local int current_instr_id_in_thread = 0;
  return &current_instr_id_in_thread;
}

inline int GetCurrentInstructionIdThisThread() {
  return *GetInstructionIdThisThreadInternal();
}

inline void SetCurrentInstructionIdThisThread(int instruction_id) {
  *GetInstructionIdThisThreadInternal() = instruction_id;
}

inline int GetNextInstructionId() {
  static int next_instruction_id = 0;
  return next_instruction_id++;
}

namespace oneflow {
class ForeignStackGetter {
 public:
  virtual ~ForeignStackGetter() = default;
  virtual void RecordCurrentStack(int64_t id) = 0;
  virtual void Print(int64_t id) const = 0;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FOREIGN_STACK_GETTER_H

