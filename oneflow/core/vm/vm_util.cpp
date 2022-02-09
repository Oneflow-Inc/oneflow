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
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {
namespace vm {

intrusive::shared_ptr<InstructionMsg> NewInstruction(const std::string& instr_type_name) {
  return intrusive::make_shared<InstructionMsg>(instr_type_name);
}

Maybe<void> Run(vm::InstructionMsgList* instr_msg_list) {
  auto* virtual_machine = JUST(GlobalMaybe<VirtualMachine>());
  JUST(virtual_machine->Receive(instr_msg_list));
  return Maybe<void>::Ok();
}

Maybe<void> ClusterSync() {
  BlockingCounter bc(1);
  JUST(PhysicalRun([&bc](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->ComputeGlobalFrontSeqBarrier());
    JUST(builder->ComputeRankFrontSeqCallback([&bc]() { bc.Decrease(); }));
    return Maybe<void>::Ok();
  }));
  JUST(bc.WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreErasedLivelyInstructions()));
  return Maybe<void>::Ok();
}

Maybe<void> CurrentRankSync() {
  BlockingCounter bc(1);
  JUST(PhysicalRun([&bc](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->ComputeRankFrontSeqCallback([&bc]() { bc.Decrease(); }));
    return Maybe<void>::Ok();
  }));
  JUST(bc.WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreErasedLivelyInstructions()));
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
