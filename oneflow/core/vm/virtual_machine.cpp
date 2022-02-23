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
#include "oneflow/core/vm/no_arg_cb_phy_instr_operand.h"
#include "oneflow/core/vm/barrier_instruction_type.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/thread/thread_consistent_id.h"
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/stream_on_independent_thread.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/common/env_var.h"
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
    JUST(DoEach(thread_ctx));
  }
  return Maybe<void>::Ok();
}

void GetSchedulerThreadInitializer(std::function<void()>* Initializer) {
  *Initializer = [&]() {
    if (!CHECK_JUST(IsMultiClient())) { return; }
    CHECK_JUST(InitThisThreadUniqueConsistentId(kThreadConsistentIdScheduler, "scheduler"));
    OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Scheduler");
  };
}

void GetCallbackThreadInitializer(std::function<void()>* Initializer) {
  *Initializer = [&]() {
    if (!CHECK_JUST(IsMultiClient())) { return; }
    OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Callback");
  };
}

void WorkerLoop(vm::ThreadCtx* thread_ctx, const std::function<void(vm::ThreadCtx*)>& Initializer) {
  Initializer(thread_ctx);
  while (thread_ctx->ReceiveAndRun() == intrusive::kChannelStatusSuccess) {}
}

}  // namespace

VirtualMachine::VirtualMachine() {
  // Class VirtualMachineEngine only cares the basic logical of vm, while class VirtualMachine
  // manages threads and condition variables.
  // In order to notify threads in VirtualMachineEngine, a notify callback lambda should be take as
  // an argument for VirtualMachineEngine's constructor.
  vm_ = intrusive::make_shared<vm::VirtualMachineEngine>([this]() { callback_notifier_.Notify(); });
  OF_PROFILER_NAME_THIS_HOST_THREAD("_Main");
  std::function<void()> CallbackInitializer;
  GetCallbackThreadInitializer(&CallbackInitializer);
  callback_thread_ = std::thread(&VirtualMachine::CallbackLoop, this, CallbackInitializer);
  std::function<void()> SchedulerInitializer;
  GetSchedulerThreadInitializer(&SchedulerInitializer);
  schedule_thread_ = std::thread(&VirtualMachine::ScheduleLoop, this, SchedulerInitializer);
  transport_local_dep_object_.Reset();
}

namespace {

Maybe<Symbol<Stream>> GetBarrierStream() {
  auto device = JUST(Device::New("control"));
  return Stream::New(device, StreamRole::kBarrier);
}

void MakeBarrierInstructions(vm::InstructionMsgList* list,
                             const std::function<void()>& ComputeCallback) {
  const auto& phy_instr_operand = std::make_shared<vm::NoArgCbPhyInstrOperand>(ComputeCallback);
  auto stream = CHECK_JUST(GetBarrierStream());
  auto instruction = intrusive::make_shared<vm::InstructionMsg>(
      stream->mut_vm_stream(), SingletonPtr<vm::BarrierInstructionType>(), phy_instr_operand);
  list->EmplaceBack(std::move(instruction));
}

}  // namespace

void VirtualMachine::ControlSync() {
  auto bc = std::make_shared<BlockingCounter>(1);
  vm::InstructionMsgList list;
  MakeBarrierInstructions(&list, [bc] { bc->Decrease(); });
  CHECK_JUST(Receive(&list));
  CHECK_JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
}

VirtualMachine::~VirtualMachine() {
  ControlSync();
  pending_notifier_.Close();
  schedule_thread_.join();
  CHECK(!vm_);
  callback_notifier_.Close();
  callback_thread_.join();
}

std::function<Maybe<bool>()> VirtualMachine::GetPredicatorNoMoreInstructionsFinished() {
  auto last_total_erased = std::make_shared<size_t>(0);
  auto* vm = Global<VirtualMachine>::Get();
  if (vm != nullptr) { *last_total_erased = vm->vm().total_erased_lively_instruction_cnt(); }
  return [last_total_erased]() -> Maybe<bool> {
    auto* vm = Global<VirtualMachine>::Get();
    CHECK_NOTNULL_OR_RETURN(vm) << "virtual machine not initialized.";
    CHECK_OR_RETURN(!vm->NoMoreErasedLivelyInstructions(last_total_erased.get()))
        << "blocking instructions\n"
        << vm->GetBlockingDebugString();
    return false;
  };
}

bool VirtualMachine::NoMoreErasedLivelyInstructions(
    size_t* last_total_erased_lively_instruction_cnt) const {
  size_t cnt = vm_->total_erased_lively_instruction_cnt();
  bool no_more_erased = (*last_total_erased_lively_instruction_cnt == cnt);
  *last_total_erased_lively_instruction_cnt = cnt;
  return no_more_erased;
}

std::string VirtualMachine::GetBlockingDebugString() {
  size_t limit = EnvInteger<ONEFLOW_VM_BLOCKING_DEBUG_INSTRUCTIONS_DISPLAY_LIMIT>();
  return vm_->GetLivelyInstructionListDebugString(limit);
}

Maybe<void> VirtualMachine::Receive(vm::InstructionMsgList* instr_list) {
  if (unlikely(pthread_fork::IsForkedSubProcess())) {
    CHECK_OR_RETURN(JUST(IsMultiClient()));
    INTRUSIVE_FOR_EACH_PTR(instr_msg, instr_list) {
      const auto& device = instr_msg->stream().device();
      CHECK_OR_RETURN(device->enum_type() == DeviceType::kCPU)
          << pthread_fork::kOfCudaNotSupportInForkedSubProcess;
    }
    JUST(vm_->Receive(instr_list));
    while (!vm_->Empty()) { vm_->Schedule(); }
    vm_->Callback();
    // no scheduler processing gc, we must do it in current thread.
    vm::InstructionMsgList garbage_msg_list;
    vm_->mut_garbage_msg_list()->MoveTo(&garbage_msg_list);
  } else {
    const int64_t kHighWaterMark = GetInstructionHighWaterMark();
    if (vm_->flying_instruction_cnt() > kHighWaterMark) {
      JUST(Global<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
        auto bc = std::make_shared<BlockingCounter>(1);
        vm_->InsertProbe([bc](vm::VirtualMachineEngine* vm) {
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

void VirtualMachine::ScheduleLoop(const std::function<void()>& Initializer) {
  Initializer();
  auto* vm = mut_vm();
  while (pending_notifier_.WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) {
    OF_PROFILER_RANGE_PUSH("VirtualMachine::ScheduleLoop");
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
        // Use ThreadUnsafeEmpty to avoid acquiring mutex lock.
        // It's safe to use ThreadUnsafeEmpty here. pending_notifier_.notified_cnt_ will be greater
        // than zero when inconsistency between
        // vm->pending_msg_list.list_head_.list_head_.container_ and
        // vm->pending_msg_list.list_head_.list_head_.size_ occured. hence the pending
        // instructions
        // will get handled in the next iteration.
        //  VirtualMachine::Receive may be less effiencient if the thread safe version `vm->Empty()`
        // used
        //  here, because VirtualMachine::ScheduleLoop is more likely to get the mutex lock.
        do { vm->Schedule(); } while (!vm->ThreadUnsafeEmpty());
        vm->NotifyCallback();
      } while (++i < kNumSchedulingPerTimoutTest);
    } while (MicrosecondsFrom(start) < kWorkingMicroseconds);
    OF_PROFILER_RANGE_POP();
  }
  while (!(vm->Empty() && vm->CallbackEmpty())) {
    vm->Schedule();
    vm->NotifyCallback();
  }
  CHECK_JUST(ForEachThreadCtx(vm_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    thread_ctx->mut_pending_instruction_list()->Close();
    return Maybe<void>::Ok();
  }));
  {
    std::unique_lock<std::mutex> lock(worker_threads_mutex_);
    for (const auto& worker_thread : worker_threads_) { worker_thread->join(); }
  }
  vm_.Reset();
}

void VirtualMachine::CallbackLoop(const std::function<void()>& Initializer) {
  Initializer();
  auto* vm = mut_vm();
  while (callback_notifier_.WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) { vm->Callback(); }
}

vm::MirroredObject* VirtualMachine:MakeScheduleLocalDepObject(Symbol<Device> device, StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  auto key = std::make_pair(device, stream_role);
  intrusive::shared_ptr<vm::MirroredObject>* ptr = &device_stream_role2local_dep_object_[key];
  if (!*ptr) { *ptr = intrusive::make_shared<vm::VirtualMachine>(); }
  return ptr->Mutable();
}

vm::MirroredObject* VirtualMachine::MakeTransportLocalDepObject() {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  if (!transport_local_dep_object_) {
    transport_local_dep_object_ = intrusive::make_shared<vm::VirtualMachine>();
  }
  return transport_local_dep_object_.Mutable();
}

Maybe<vm::Stream*> VirtualMachine::CreateStream(Symbol<Device> device, StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  vm::ThreadCtx* thread_ctx = JUST(FindOrCreateThreadCtx(device, stream_role)); 
  return JUST(CreateStream(thread_ctx, device, stream_role));
}

Maybe<vm::ThreadCtx*> VirtualMachine::FindOrCreateThreadCtx(
    Symbol<Device> device, StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  vm::ThreadCtx** thread_ctx_ptr = nullptr;
  if (StreamRoleSwitch<StreamOnIndependentThread>(stream_role)) {
    auto key = std::make_pair(device->enum_type(), stream_role);
    thread_ctx_ptr = &devcie_type_stream_role_2independent_thread_ctx_[key];
  } else {
    thread_ctx_ptr = devcie_type2non_independent_thread_ctx_[device->enum_type()];
  }
  if (*thread_ctx_ptr == nullptr) {
    *thread_ctx_ptr = JUST(CreateThreadCtx(device, stream_role));
  }
  return *thread_ctx_ptr;
}

Maybe<vm::ThreadCtx*> VirtualMachine::CreateThreadCtx(Symbol<Device> device, StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  // thread_ctx_ptr may be used after timout.
  auto thread_ctx_ptr = std::make_shared<vm::ThreadCtx*>(nullptr);
  {
    auto bc = std::make_shared<BlockingCounter>(1);
    vm_->InsertProbe([thread_ctx_ptr, bc](vm::VirtualMachineEngine* vm) {
      auto thread_ctx = intrusive::make_shared<vm::ThreadCtx>();
      vm->mut_thread_ctx_list()->PushBack(thread_ctx.Mutable());
      *thread_ctx_ptr = thread_ctx.Mutable();
      bc->Decrease();
      return true;
    });
    pending_notifier_.Notify();
    JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  }
  auto* thread_ctx = *thread_ctx_ptr;
  {
    const auto& WorkerInitializer = [device, stream_role, thread_ctx](vm::ThreadCtx* thread_ctx) {
      int device_type_value = static_cast<int>(device->enum_type());
      CHECK_GT(device_type_value, 0);
      std::string device_tag = *CHECK_JUST(DeviceTag4DeviceType(device->enum_type()));
      if (!StreamRoleSwitch<StreamOnIndependentThread>(stream_role)) {
        CHECK_JUST(InitThisThreadConsistentId(device_type_value + kThreadConsistentIdScheduler,
                                              device_tag));
      }
      OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Worker_" + device_tag);
    };
    auto thread = std::make_unique<std::thread>(&WorkerLoop, thread_ctx, WorkerInitializer);
    std::unique_lock<std::mutex> lock(worker_threads_mutex_);
    worker_threads_.push_back(std::move(thread));
  }
  return thread_ctx;
}

Maybe<vm::Stream*> VirtualMachine::CreateStream(vm::ThreadCtx* thread_ctx, Symbol<Device> device, StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  // stream_ptr may be used after timout.
  auto stream_ptr = std::make_shared<vm::Stream*>(nullptr);
  auto bc = std::make_shared<BlockingCounter>(1);
  vm_->InsertProbe([stream_ptr, thread_ctx, bc](vm::VirtualMachineEngine* vm) {
    auto stream = intrusive::make_shared<vm::Stream>(thread_ctx, device, stream_role);
    thread_ctx->mut_stream_list()->PushBack(stream.Mutable());
    *stream_ptr = stream.Mutable();
    bc->Decrease();
    return true;
  });
  pending_notifier_.Notify();
  JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return *stream_ptr;
}

}  // namespace oneflow
