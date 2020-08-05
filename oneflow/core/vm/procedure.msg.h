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
#ifndef ONEFLOW_CORE_VM_PROCEDURE_H_
#define ONEFLOW_CORE_VM_PROCEDURE_H_

#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(Procedure);
  // fields
  OBJECT_MSG_DEFINE_STRUCT(ParallelConf, parallel_conf);
  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(InstructionMsg, instr_msg_link, local_instr_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(InstructionMsg, instr_msg_link, remote_instr_list);
OBJECT_MSG_END(Procedure);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_PROCEDURE_H_
