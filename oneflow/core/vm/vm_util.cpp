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
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {
namespace vm {

Maybe<void> Run(vm::InstructionList* instruction_list) {
  auto* virtual_machine = JUST(SingletonMaybe<VirtualMachine>());
  JUST(virtual_machine->Receive(instruction_list));
  return Maybe<void>::Ok();
}

Maybe<void> ClusterSync() {
  auto bc = std::make_shared<BlockingCounter>(1);
  JUST(PhysicalRun([bc](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->GlobalSync());
    JUST(builder->Barrier([bc]() { bc->Decrease(); }));
    return Maybe<void>::Ok();
  }));
  JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return Maybe<void>::Ok();
}

Maybe<void> CurrentRankSync() {
  auto bc = std::make_shared<BlockingCounter>(1);
  JUST(PhysicalRun([bc](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->Barrier([bc]() { bc->Decrease(); }));
    return Maybe<void>::Ok();
  }));
  JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
