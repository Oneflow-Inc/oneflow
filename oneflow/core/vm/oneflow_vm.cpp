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
#include <typeinfo>
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/no_arg_cb_phy_instr_operand.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/thread/thread_consistent_id.h"
#include "oneflow/core/framework/transport_token.h"

namespace oneflow {

namespace {

Maybe<void> ForEachThreadCtx(vm::VirtualMachine* vm,
                             const std::function<Maybe<void>(vm::ThreadCtx*)>& DoEach) {
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(thread_ctx, vm->mut_thread_ctx_list()) {
    const auto& stream_type = thread_ctx->stream_rt_desc().stream_type_id().stream_type();
    if (stream_type.SharingVirtualMachineThread()) { continue; }
    JUST(DoEach(thread_ctx));
  }
  return Maybe<void>::Ok();
}

void GetSchedulerThreadInitializer(std::function<void()>* Initializer) {
  *Initializer = [&]() {
    if (!CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) { return; }
    CHECK_JUST(InitThisThreadUniqueConsistentId(kThreadConsistentIdScheduler, "scheduler"));
  };
}

std::type_index GetStreamTypeIndex(const vm::ThreadCtx* thread_ctx) {
  const auto& stream_rt_desc = thread_ctx->stream_rt_desc();
  const auto& stream_type_id = stream_rt_desc.stream_type_id();
  const auto& stream_type = stream_type_id.stream_type();
  return typeid(stream_type);
}

// Threads with the same stream_type share a thread_consistent_id.
// e.g.
//   Given there are 8 gpu thread in a single process.
//   thread #0 is active in process #0, while others are not.
//   thread #1 is active in process #1, while others are not.
//   ...
//   thread #7 is active in process #7, while others are not.
//   to make them communicate with each other, we can allocate thread_consistent_id 1 to all those
//   gpu threads in all processes.
void GetWorkerThreadInitializer(intrusive::shared_ptr<vm::VirtualMachine> vm,
                                std::function<void(vm::ThreadCtx*)>* Initializer) {
  std::set<std::type_index> stream_type_indexes;
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(thread_ctx, vm->mut_thread_ctx_list()) {
    const auto& stream_type = thread_ctx->stream_rt_desc().stream_type_id().stream_type();
    if (!stream_type.SupportingTransportInstructions()) { continue; }
    stream_type_indexes.insert(GetStreamTypeIndex(thread_ctx));
  }
  HashMap<std::type_index, int64_t> stream_type_index2consistent_id;
  int64_t thread_consistent_id = kThreadConsistentIdScheduler + 1;
  for (const auto& stream_type_index : stream_type_indexes) {
    LOG(INFO) << "transport stream type: " << stream_type_index.name();
    stream_type_index2consistent_id[stream_type_index] = thread_consistent_id++;
  }
  *Initializer = [stream_type_index2consistent_id](vm::ThreadCtx* thread_ctx) {
    if (!CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) { return; }
    const auto& stream_type_index = GetStreamTypeIndex(thread_ctx);
    const auto& iter = stream_type_index2consistent_id.find(stream_type_index);
    if (iter != stream_type_index2consistent_id.end()) {
      CHECK_JUST(InitThisThreadConsistentId(iter->second, stream_type_index.name()));
    }
  };
}

}  // namespace

OneflowVM::OneflowVM(const Resource& resource, int64_t this_machine_id)
    : vm_(intrusive::make_shared<vm::VirtualMachine>(
        vm::MakeVmDesc(resource, this_machine_id).Get())) {
  std::function<void()> SchedulerInitializer;
  GetSchedulerThreadInitializer(&SchedulerInitializer);
  std::function<void(vm::ThreadCtx*)> WorkerInitializer;
  GetWorkerThreadInitializer(vm_, &WorkerInitializer);
  CHECK_JUST(ForEachThreadCtx(vm_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    auto thread =
        std::make_unique<std::thread>(&vm::ThreadCtx::LoopRun, thread_ctx, WorkerInitializer);
    worker_threads_.push_back(std::move(thread));
    return Maybe<void>::Ok();
  }));
  schedule_thread_ = std::thread(&OneflowVM::Loop, this, SchedulerInitializer);
}

namespace {

void MakeCtrlSeqInstructions(vm::InstructionMsgList* list,
                             const std::function<void()>& ComputeCallback) {
  auto instruction = vm::NewInstruction("CtrlComputeRankFrontSeqCallback");
  instruction->add_int64_operand(GlobalProcessCtx::Rank());
  *instruction->mut_phy_instr_operand() =
      std::make_shared<vm::NoArgCbPhyInstrOperand>(ComputeCallback);
  list->EmplaceBack(std::move(instruction));
}

}  // namespace

void OneflowVM::ControlSync() {
  BlockingCounter bc(1);
  vm::InstructionMsgList list;
  MakeCtrlSeqInstructions(&list, [&] { bc.Decrease(); });
  CHECK_JUST(Receive(&list));
  bc.WaitUntilCntEqualZero();
}

OneflowVM::~OneflowVM() {
  ControlSync();
  notifier_.Close();
  schedule_thread_.join();
  CHECK(!vm_);
}

Maybe<void> OneflowVM::Receive(vm::InstructionMsgList* instr_list) {
  JUST(vm_->Receive(instr_list));
  notifier_.Notify();
  return Maybe<void>::Ok();
}

void OneflowVM::Loop(const std::function<void()>& Initializer) {
  Initializer();
  auto* vm = mut_vm();
  while (notifier_.WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) {
    // Use ThreadUnsafeEmpty to avoid acquiring mutex lock.
    // It's safe to use ThreadUnsafeEmpty here. notifier_.notified_cnt_ will be greater than zero
    // when inconsistency between vm->pending_msg_list.list_head_.list_head_.container_ and
    // vm->pending_msg_list.list_head_.list_head_.size_ occured. hence the pending instructions will
    // get handled in the next iteration.
    //  OneflowVM::Receive may be less effiencient if the thread safe version `vm->Empty()` used
    //  here, because OneflowVM::Loop is more likely to get the mutex lock.
    while (!vm->ThreadUnsafeEmpty()) { vm->Schedule(); }
  }
  while (!vm->Empty()) { vm->Schedule(); }
  CHECK_JUST(ForEachThreadCtx(vm_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    thread_ctx->mut_pending_instruction_list()->Close();
    return Maybe<void>::Ok();
  }));
  for (const auto& worker_thread : worker_threads_) { worker_thread->join(); }
  vm_.Reset();
}

}  // namespace oneflow
