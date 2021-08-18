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

namespace {

Maybe<void> ForEachThreadCtx(vm::VirtualMachine* vm,
                             const std::function<Maybe<void>(vm::ThreadCtx*)>& DoEach) {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm->mut_thread_ctx_list(), thread_ctx) {
    const auto& stream_type = thread_ctx->stream_rt_desc().stream_type_id().stream_type();
    if (stream_type.SharingVirtualMachineThread()) { continue; }
    JUST(DoEach(thread_ctx));
  }
  return Maybe<void>::Ok();
}

}  // namespace

OneflowVM::OneflowVM(const Resource& resource, int64_t this_machine_id)
    : vm_(ObjectMsgPtr<vm::VirtualMachine>::New(vm::MakeVmDesc(resource, this_machine_id).Get())) {
  CHECK_JUST(ForEachThreadCtx(vm_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    auto thread = std::make_unique<std::thread>(&vm::ThreadCtx::LoopRun, thread_ctx);
    worker_threads_.push_back(std::move(thread));
    return Maybe<void>::Ok();
  }));
  exiting_ = false;
  schedule_thread_ = std::thread(&OneflowVM::Loop, this);
}

namespace {

void MakeCtrlSeqInstructions(vm::InstructionMsgList* list,
                             const std::function<void()>& ComputeCallback) {
  auto instruction = vm::NewInstruction("CtrlComputeRankFrontSeqCallback");
  instruction->add_int64_operand(GlobalProcessCtx::Rank());
  *instruction->mutable_phy_instr_operand() =
      std::make_shared<vm::NoArgCbPhyInstrOperand>(ComputeCallback);
  list->EmplaceBack(std::move(instruction));
}

void ControlSync(vm::VirtualMachine* vm) {
  BlockingCounter bc(1);
  vm::InstructionMsgList list;
  MakeCtrlSeqInstructions(&list, [&] { bc.Decrease(); });
  CHECK_JUST(vm->Receive(&list));
  bc.WaitUntilCntEqualZero();
}

}  // namespace

OneflowVM::~OneflowVM() {
  ControlSync(mut_vm());
  exiting_ = true;
  schedule_thread_.join();
  CHECK(!vm_);
}

void OneflowVM::Loop() {
  auto* vm = mut_vm();
  while (!exiting_) { vm->Schedule(); }
  while (!mut_vm()->Empty()) { vm->Schedule(); }
  CHECK_JUST(ForEachThreadCtx(vm_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    thread_ctx->mut_pending_instruction_list()->Close();
    return Maybe<void>::Ok();
  }));
  for (const auto& worker_thread : worker_threads_) { worker_thread->join(); }
  vm_.Reset();
}

}  // namespace oneflow
