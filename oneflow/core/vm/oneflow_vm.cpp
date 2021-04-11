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
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/no_arg_cb_phy_instr_operand.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

OneflowVM::OneflowVM(const Resource& resource, int64_t this_machine_id)
    : vm_(ObjectMsgPtr<vm::VirtualMachine>::New(vm::MakeVmDesc(resource, this_machine_id).Get())) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_->mut_thread_ctx_list(), thread_ctx) {
    auto thread = std::make_unique<std::thread>(&vm::ThreadCtx::LoopRun, thread_ctx);
    worker_threads_.push_back(std::move(thread));
  }
  exiting_ = false;
  scheduler_exited_ = false;
  schedule_thread_ = std::thread(&OneflowVM::Loop, this);
}

namespace {

void MakeCtrlSeqInstructions(vm::InstructionMsgList* list,
                             const std::function<void()>& InferCallback,
                             const std::function<void()>& ComputeCallback) {
  {
    auto instruction = vm::NewInstruction("CtrlInferRankFrontSeqCallback");
    instruction->add_int64_operand(GlobalProcessCtx::Rank());
    *instruction->mutable_phy_instr_operand() =
        std::make_shared<vm::NoArgCbPhyInstrOperand>(InferCallback);
    list->EmplaceBack(std::move(instruction));
  }
  {
    auto instruction = vm::NewInstruction("CtrlComputeRankFrontSeqCallback");
    instruction->add_int64_operand(GlobalProcessCtx::Rank());
    *instruction->mutable_phy_instr_operand() =
        std::make_shared<vm::NoArgCbPhyInstrOperand>(ComputeCallback);
    list->EmplaceBack(std::move(instruction));
  }
}

void ControlSync(vm::VirtualMachine* vm) {
  BlockingCounter bc(2);
  vm::InstructionMsgList list;
  MakeCtrlSeqInstructions(
      &list, [&] { bc.Decrease(); }, [&] { bc.Decrease(); });
  vm->Receive(&list);
  bc.WaitUntilCntEqualZero();
}

}  // namespace

OneflowVM::~OneflowVM() {
  ControlSync(mut_vm());
  exiting_ = true;
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_->mut_thread_ctx_list(), thread_ctx) {
    thread_ctx->mut_pending_instruction_list()->Close();
  }
  for (const auto& worker_thread : worker_threads_) { worker_thread->join(); }
  schedule_thread_.join();
  CHECK(scheduler_exited_);
  CHECK(mut_vm()->Empty());
}

void OneflowVM::Loop() {
  auto* vm = mut_vm();
  while (!exiting_) { vm->Schedule(); }
  scheduler_exited_ = true;
}

}  // namespace oneflow
