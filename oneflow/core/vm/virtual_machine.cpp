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
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/barrier_phy_instr_operand.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/thread/thread_consistent_id.h"
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

namespace {

template<typename T>
int MicrosecondsFrom(const T& start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()
                                                               - start)
      .count();
}

Maybe<void> ForEachThreadCtx(vm::VirtualMachineEngine* vm,
                             const std::function<Maybe<void>(vm::ThreadCtx*)>& DoEach) {
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(thread_ctx, vm->mut_thread_ctx_list()) {
    const auto& stream_type = thread_ctx->stream_rt_desc().stream_type();
    if (stream_type.OnSchedulerThread()) { continue; }
    JUST(DoEach(thread_ctx));
  }
  return Maybe<void>::Ok();
}

void GetSchedulerThreadInitializer(std::function<void()>* Initializer) {
  *Initializer = [&]() {
    CHECK_JUST(InitThisThreadUniqueConsistentId(kThreadConsistentIdScheduler, "scheduler"));
    OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Scheduler");
  };
}

void GetCallbackThreadInitializer(std::function<void()>* Initializer) {
  *Initializer = [&]() { OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Callback"); };
}

std::type_index GetStreamTypeIndex(const vm::ThreadCtx* thread_ctx) {
  const auto& stream_rt_desc = thread_ctx->stream_rt_desc();
  const auto& stream_type = stream_rt_desc.stream_type();
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
void GetWorkerThreadInitializer(intrusive::shared_ptr<vm::VirtualMachineEngine> vm,
                                std::function<void(vm::ThreadCtx*)>* Initializer) {
  std::set<std::type_index> stream_type_indexes;
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(thread_ctx, vm->mut_thread_ctx_list()) {
    const auto& stream_type = thread_ctx->stream_rt_desc().stream_type();
    if (!stream_type.SupportingTransportInstructions()) { continue; }
    stream_type_indexes.insert(GetStreamTypeIndex(thread_ctx));
  }
  HashMap<std::type_index, int64_t> stream_type_index2consistent_id;
  int64_t thread_consistent_id = kThreadConsistentIdScheduler + 1;
  for (const auto& stream_type_index : stream_type_indexes) {
    VLOG(3) << "transport stream type: " << stream_type_index.name();
    stream_type_index2consistent_id[stream_type_index] = thread_consistent_id++;
  }
  *Initializer = [stream_type_index2consistent_id](vm::ThreadCtx* thread_ctx) {
    const auto& stream_type_index = GetStreamTypeIndex(thread_ctx);
    const auto& iter = stream_type_index2consistent_id.find(stream_type_index);
    if (iter != stream_type_index2consistent_id.end()) {
      CHECK_JUST(InitThisThreadConsistentId(iter->second, stream_type_index.name()));
    }
    OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Worker");
  };
}

void WorkerLoop(vm::ThreadCtx* thread_ctx, const std::function<void(vm::ThreadCtx*)>& Initializer) {
  Initializer(thread_ctx);
  while (thread_ctx->mut_notifier()->WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) {
    while (thread_ctx->TryReceiveAndRun()) {}
  }
}

}  // namespace

VirtualMachine::VirtualMachine(const Resource& resource, int64_t this_machine_id)
    : vm_threads_closed_(false) {
  // Class VirtualMachineEngine only cares the basic logical of vm, while class VirtualMachine
  // manages threads and condition variables.
  // In order to notify threads in VirtualMachineEngine, a notify callback lambda should be take as
  // an argument for VirtualMachineEngine's constructor.
  vm_ = intrusive::make_shared<vm::VirtualMachineEngine>(
      vm::MakeVmDesc(resource, this_machine_id).Get());
  OF_PROFILER_NAME_THIS_HOST_THREAD("_Main");
  std::function<void(vm::ThreadCtx*)> WorkerInitializer;
  GetWorkerThreadInitializer(vm_, &WorkerInitializer);
  CHECK_JUST(ForEachThreadCtx(vm_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    auto thread = std::make_unique<std::thread>(&WorkerLoop, thread_ctx, WorkerInitializer);
    worker_threads_.push_back(std::move(thread));
    return Maybe<void>::Ok();
  }));
  std::function<void()> CallbackInitializer;
  GetCallbackThreadInitializer(&CallbackInitializer);
  callback_thread_ = std::thread(&VirtualMachine::CallbackLoop, this, CallbackInitializer);
  std::function<void()> SchedulerInitializer;
  GetSchedulerThreadInitializer(&SchedulerInitializer);
  schedule_thread_ = std::thread(&VirtualMachine::ScheduleLoop, this, SchedulerInitializer);
}

namespace {

void MakeCtrlSeqInstructions(vm::VirtualMachineEngine* vm, vm::InstructionMsgList* list,
                             const std::function<void()>& ComputeCallback) {
  const auto& phy_instr_operand = std::make_shared<vm::BarrierPhyInstrOperand>(ComputeCallback);
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      vm, "CtrlComputeRankFrontSeqCallback", std::shared_ptr<const ParallelDesc>(),
      phy_instr_operand);
  list->EmplaceBack(std::move(instruction));
}

}  // namespace

void VirtualMachine::ControlSync() {
  auto bc = std::make_shared<BlockingCounter>(1);
  vm::InstructionMsgList list;
  MakeCtrlSeqInstructions(mut_vm(), &list, [bc] { bc->Decrease(); });
  CHECK_JUST(Receive(&list));
  CHECK_JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
}

Maybe<void> VirtualMachine::CloseVMThreads() {
  CHECK_OR_RETURN(!vm_threads_closed_);
  ControlSync();
  pending_notifier_.Close();
  schedule_thread_.join();
  vm_threads_closed_ = true;
  return Maybe<void>::Ok();
}

VirtualMachine::~VirtualMachine() {
  if (!vm_threads_closed_) { CHECK_JUST(CloseVMThreads()); }
  CHECK(vm_->SchedulerEmpty());
  CHECK(vm_->CallbackEmpty());
  vm_.Reset();
  callback_notifier_.Close();
  callback_thread_.join();
}

std::function<Maybe<bool>()> VirtualMachine::GetPredicatorNoMoreInstructionsFinished() {
  auto last_total_erased = std::make_shared<size_t>(0);
  auto* vm = Global<VirtualMachine>::Get();
  if (vm != nullptr) { *last_total_erased = vm->vm().total_erased_instruction_cnt(); }
  return [last_total_erased]() -> Maybe<bool> {
    auto* vm = Global<VirtualMachine>::Get();
    CHECK_NOTNULL_OR_RETURN(vm) << "virtual machine not initialized.";
    CHECK_OR_RETURN(!vm->NoMoreErasedInstructions(last_total_erased.get()))
        << "blocking instructions\n"
        << vm->GetBlockingDebugString();
    return false;
  };
}

bool VirtualMachine::NoMoreErasedInstructions(size_t* last_total_erased_instruction_cnt) const {
  size_t cnt = vm_->total_erased_instruction_cnt();
  bool no_more_erased = (*last_total_erased_instruction_cnt == cnt);
  *last_total_erased_instruction_cnt = cnt;
  return no_more_erased;
}

std::string VirtualMachine::GetBlockingDebugString() {
  size_t limit = EnvInteger<ONEFLOW_VM_BLOCKING_DEBUG_INSTRUCTIONS_DISPLAY_LIMIT>();
  return vm_->GetLivelyInstructionListDebugString(limit);
}

Maybe<void> VirtualMachine::Receive(vm::InstructionMsgList* instr_list) {
  if (unlikely(pthread_fork::IsForkedSubProcess())) {
    INTRUSIVE_FOR_EACH_PTR(instr_msg, instr_list) {
      const auto& parallel_desc = instr_msg->phy_instr_parallel_desc();
      CHECK_OR_RETURN(!parallel_desc || parallel_desc->device_type() == DeviceType::kCPU)
          << pthread_fork::kOfCudaNotSupportInForkedSubProcess;
      // NOTE: operate `vm_` in forked subprocesses causes mysterious problems.
      // `ComputeInFuseMode` will be replaced by `Compute` soon.
      instr_msg->mut_instr_type_id()->instruction_type().ComputeInFuseMode(instr_msg);
    }
  } else if (unlikely(vm_threads_closed_)) {
    JUST(RunInCurrentThread(instr_list));
  } else {
    const int64_t kHighWaterMark = GetInstructionHighWaterMark();
    if (vm_->flying_instruction_cnt() > kHighWaterMark) {
      JUST(Global<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
        auto bc = std::make_shared<BlockingCounter>(1);
        vm_->InsertCallbackProbe([bc](vm::VirtualMachineEngine* vm) {
          const int64_t kLowWaterMark = GetInstructionLowWaterMark();
          if (vm->flying_instruction_cnt() > kLowWaterMark) { return false; }
          bc->Decrease();
          return true;
        });
        pending_notifier_.Notify();
        JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
        return Maybe<void>::Ok();
      }));
    }
    if (JUST(vm_->Receive(instr_list))) {
      // old pending_instruction_list is empty.
      pending_notifier_.Notify();
    }
  }
  return Maybe<void>::Ok();
}

namespace {

class SingleThreadScheduleCtx : public vm::ScheduleCtx {
 public:
  explicit SingleThreadScheduleCtx(vm::VirtualMachineEngine* vm) : vm_(vm) {}
  ~SingleThreadScheduleCtx() = default;

  void OnGarbageMsgPending() const override { vm_->Callback(); }
  void OnWorkerLoadPending(vm::ThreadCtx* thread_ctx) const override {
    while (thread_ctx->TryReceiveAndRun() > 0) {}
  }

 private:
  vm::VirtualMachineEngine* vm_;
};

void ScheduleUntilVMEmpty(vm::VirtualMachineEngine* vm, const vm::ScheduleCtx& schedule_ctx) {
  do {
    vm->Schedule(schedule_ctx);
    vm->FlushGarbageInstructions(schedule_ctx);
  } while (!(vm->SchedulerEmpty() && vm->CallbackEmpty()));
}

}  // namespace

Maybe<void> VirtualMachine::RunInCurrentThread(vm::InstructionMsgList* instr_list) {
  CHECK_OR_RETURN(vm_->SchedulerEmpty()) << "vm scheduler not empty. May be a fatal error occured";
  CHECK_OR_RETURN(vm_->CallbackEmpty())
      << "vm callback handler not empty. May be a fatal error occured";
  JUST(vm_->Receive(instr_list));
  ScheduleUntilVMEmpty(vm_.Mutable(), SingleThreadScheduleCtx(vm_.Mutable()));
  return Maybe<void>::Ok();
}

namespace {

class MultiThreadScheduleCtx : public vm::ScheduleCtx {
 public:
  explicit MultiThreadScheduleCtx(Notifier* cb_notifier) : cb_notifier_(cb_notifier) {}
  ~MultiThreadScheduleCtx() = default;

  void OnGarbageMsgPending() const override { cb_notifier_->Notify(); }
  void OnWorkerLoadPending(vm::ThreadCtx* thread_ctx) const override {
    thread_ctx->mut_notifier()->Notify();
  }

 private:
  Notifier* cb_notifier_;
};

}  // namespace

void VirtualMachine::ScheduleLoop(const std::function<void()>& Initializer) {
  Initializer();
  MultiThreadScheduleCtx schedule_ctx(&callback_notifier_);
  auto* vm = mut_vm();
  while (pending_notifier_.WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) {
    OF_PROFILER_RANGE_GUARD("VirtualMachine::ScheduleLoop");
    auto start = std::chrono::steady_clock::now();
    static constexpr int kWorkingMicroseconds = 1000;
    // Every time this thread wakes up, vm is scheduled for about `kWorkingMicroseconds`.
    // The cost of os thread switching is about 5-10 microseconds. Doing more scheduling in
    // a single waiting up can reach higher performance.
    do {
      static constexpr int kNumSchedulingPerTimoutTest = 10000;
      // Every time kWorkingMicroseconds timeout tested, vm is scheduled for about
      // kNumSchedulingPerTimoutTest.
      // The cost of `MicrosecondsFrom(start)` is about 400ns, while the empty scheduling costs
      // about 10ns.
      int i = 0;
      do {
        // Use SchedulerThreadUnsafeEmpty to avoid acquiring mutex lock.
        // It's safe to use SchedulerThreadUnsafeEmpty here. pending_notifier_.notified_cnt_ will be
        // greater than zero when inconsistency between
        // vm->pending_msg_list.list_head_.list_head_.container_ and
        // vm->pending_msg_list.list_head_.list_head_.size_ occured. hence the pending
        // instructions
        // will get handled in the next iteration.
        //  VirtualMachine::Receive may be less effiencient if the thread safe version
        //  `vm->SchedulerEmpty()`
        // used
        //  here, because VirtualMachine::ScheduleLoop is more likely to get the mutex lock.
        do { vm->Schedule(schedule_ctx); } while (!vm->SchedulerThreadUnsafeEmpty());
        vm->FlushGarbageInstructions(schedule_ctx);
      } while (++i < kNumSchedulingPerTimoutTest);
    } while (MicrosecondsFrom(start) < kWorkingMicroseconds);
  }
  ScheduleUntilVMEmpty(vm, schedule_ctx);
  CHECK_JUST(ForEachThreadCtx(vm_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    thread_ctx->mut_notifier()->Close();
    return Maybe<void>::Ok();
  }));
  for (const auto& worker_thread : worker_threads_) { worker_thread->join(); }
}

void VirtualMachine::CallbackLoop(const std::function<void()>& Initializer) {
  Initializer();
  auto* vm = mut_vm();
  while (callback_notifier_.WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) { vm->Callback(); }
}

}  // namespace oneflow
