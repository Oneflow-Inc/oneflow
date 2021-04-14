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
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {
namespace vm {

ObjectMsgPtr<InstructionMsg> NewInstruction(const std::string& instr_type_name) {
  return ObjectMsgPtr<InstructionMsg>::New(instr_type_name);
}

Maybe<void> Run(vm::InstructionMsgList* instr_msg_list) {
  auto* oneflow_vm = JUST(GlobalMaybe<OneflowVM>());
  auto* vm = oneflow_vm->mut_vm();
  vm->Receive(instr_msg_list);
  return Maybe<void>::Ok();
}

Maybe<void> SingleClientSync() {
  BlockingCounter bc(1);
  LogicalRun([&bc](const std::shared_ptr<InstructionsBuilder>& builder) {
    builder->ComputeGlobalFrontSeqBarrier();
    builder->ComputeRankFrontSeqCallback([&bc]() { bc.Decrease(); });
  });

  bc.WaitUntilCntEqualZero();

  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
