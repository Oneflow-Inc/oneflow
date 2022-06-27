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
#include "oneflow/core/vm/barrier_instruction_type.h"
#include "oneflow/core/vm/barrier_phy_instr_operand.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/vm/shrinkable_cache.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/thread/thread_consistent_id.h"
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/stream_on_independent_thread.h"
#include "oneflow/core/framework/stream_is_comm_net_stream.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/stream_mgr.h"

namespace oneflow {

namespace {

template<typename T>
int MicrosecondsFrom(const T& start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()
                                                               - start)
      .count();
}

Maybe<void> ForEachThreadCtx(vm::VirtualMachineEngine* engine,
                             const std::function<Maybe<void>(vm::ThreadCtx*)>& DoEach) {
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(thread_ctx, engine->mut_thread_ctx_list()) {
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

void WorkerLoop(vm::ThreadCtx* thread_ctx, const std::function<void(vm::ThreadCtx*)>& Initializer) {
  Initializer(thread_ctx);
  while (thread_ctx->mut_notifier()->WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) {
    while (thread_ctx->TryReceiveAndRun()) {}
  }
}

}  // namespace

VirtualMachine::VirtualMachine() : disable_vm_threads_(false), scheduler_stopped_(false) {
  // Class VirtualMachineEngine only cares the basic logical of vm, while class VirtualMachine
  // manages threads and condition variables.
  // In order to notify threads in VirtualMachineEngine, a notify callback lambda should be take as
  // an argument for VirtualMachineEngine's constructor.
  engine_ = intrusive::make_shared<vm::VirtualMachineEngine>();
  OF_PROFILER_NAME_THIS_HOST_THREAD("_Main");
  std::function<void()> SchedulerInitializer;
  GetSchedulerThreadInitializer(&SchedulerInitializer);
  schedule_thread_ = std::thread(&VirtualMachine::ScheduleLoop, this, SchedulerInitializer);
  transport_local_dep_object_.Reset();
}

namespace {

Maybe<Symbol<Stream>> GetBarrierStream() {
  auto device = JUST(Device::New("cpu"));
  return Stream::New(device, StreamRole::kBarrier);
}

void MakeBarrierInstructions(vm::InstructionList* list,
                             const std::function<void()>& BarrierCallback) {
  auto* vm = Global<VirtualMachine>::Get();
  {
    const auto& phy_instr_operand = std::make_shared<vm::BarrierPhyInstrOperand>([]() {});
    auto stream = CHECK_JUST(GetBarrierStream());
    auto instruction = intrusive::make_shared<vm::Instruction>(
        CHECK_JUST(vm->GetVmStream(stream)), SingletonPtr<vm::GlobalSyncInstructionType>(),
        phy_instr_operand);
    list->EmplaceBack(std::move(instruction));
  }
  {
    const auto& phy_instr_operand = std::make_shared<vm::BarrierPhyInstrOperand>(BarrierCallback);
    auto stream = CHECK_JUST(GetBarrierStream());
    auto instruction = intrusive::make_shared<vm::Instruction>(
        CHECK_JUST(vm->GetVmStream(stream)), SingletonPtr<vm::BarrierInstructionType>(),
        phy_instr_operand);
    list->EmplaceBack(std::move(instruction));
  }
}

}  // namespace

void VirtualMachine::ControlSync() {
  auto bc = std::make_shared<BlockingCounter>(1);
  vm::InstructionList list;
  MakeBarrierInstructions(&list, [bc] { bc->Decrease(); });
  CHECK_JUST(Receive(&list));
  CHECK_JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
}

Maybe<void> VirtualMachine::CloseVMThreads() {
  CHECK_OR_RETURN(!disable_vm_threads_) << "vm threads closed";
  ControlSync();
  pending_notifier_.Close();
  schedule_thread_.join();
  disable_vm_threads_ = true;
  return Maybe<void>::Ok();
}

namespace {

class SingleThreadScheduleCtx : public vm::ScheduleCtx {
 public:
  SingleThreadScheduleCtx() = default;
  ~SingleThreadScheduleCtx() = default;

  void OnWorkerLoadPending(vm::ThreadCtx* thread_ctx) const override {
    while (thread_ctx->TryReceiveAndRun() > 0) {}
  }
};

void ScheduleUntilVMEmpty(vm::VirtualMachineEngine* vm, const vm::ScheduleCtx& schedule_ctx) {
  do { vm->Schedule(schedule_ctx); } while (!(vm->SchedulerEmpty()));
}

}  // namespace

Maybe<void> VirtualMachine::BlockingRunProbeFunc(
    const std::function<bool(vm::VirtualMachineEngine*)>& prob_func) {
  JUST(Global<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
    auto bc = std::make_shared<BlockingCounter>(1);
    engine_->InsertProbe([bc, prob_func](vm::VirtualMachineEngine* engine) {
      if (!prob_func(engine)) { return false; }
      bc->Decrease();
      return true;
    });
    if (disable_vm_threads_) {
      ScheduleUntilVMEmpty(engine_.Mutable(), SingleThreadScheduleCtx());
    } else {
      pending_notifier_.Notify();
    }
    JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> VirtualMachine::ShrinkAllMem() {
  auto try_shrink_men = [](vm::VirtualMachineEngine* engine) -> bool {
    if (engine->mut_active_stream_list()->size()) { return false; }
    INTRUSIVE_FOR_EACH_PTR(thread_ctx, engine->mut_thread_ctx_list()) {
      INTRUSIVE_FOR_EACH_PTR(stream, thread_ctx->mut_stream_list()) {
        const auto& device_ctx = stream->device_ctx();
        if (device_ctx.get() && device_ctx->mut_allocator()) {
          auto* allocator = device_ctx->mut_allocator();
          auto* cache = dynamic_cast<vm::ShrinkableCache*>(allocator);
          if (cache != nullptr) { cache->Shrink(); }
        }
      }
    }
    return true;
  };
  return BlockingRunProbeFunc(try_shrink_men);
}

VirtualMachine::~VirtualMachine() {
  if (!disable_vm_threads_) { CHECK_JUST(CloseVMThreads()); }
  CHECK(engine_->SchedulerEmpty());
  engine_.Reset();
}

std::function<Maybe<bool>()> VirtualMachine::GetPredicatorNoMoreInstructionsFinished() {
  auto last_total_erased = std::make_shared<size_t>(0);
  auto* vm = Global<VirtualMachine>::Get();
  if (vm != nullptr) { *last_total_erased = vm->engine_->total_erased_instruction_cnt(); }
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
  size_t cnt = engine_->total_erased_instruction_cnt();
  bool no_more_erased = (*last_total_erased_instruction_cnt == cnt);
  *last_total_erased_instruction_cnt = cnt;
  return no_more_erased;
}

std::string VirtualMachine::GetBlockingDebugString() {
  size_t limit = EnvInteger<ONEFLOW_VM_BLOCKING_DEBUG_INSTRUCTIONS_DISPLAY_LIMIT>();
  return engine_->GetLivelyInstructionListDebugString(limit);
}

Maybe<void> VirtualMachine::Receive(vm::InstructionList* instruction_list) {
  if (unlikely(pthread_fork::IsForkedSubProcess())) {
    INTRUSIVE_FOR_EACH_PTR(instruction, instruction_list) {
      const auto& device = instruction->stream().device();
      CHECK_OR_RETURN(device->enum_type() == DeviceType::kCPU)
          << pthread_fork::kOfCudaNotSupportInForkedSubProcess;
      instruction->instruction_type().Compute(instruction);
    }
  } else if (unlikely(disable_vm_threads_)) {
    JUST(RunInCurrentThread(instruction_list));
  } else {
    const int64_t kHighWaterMark = GetInstructionHighWaterMark();
    if (engine_->flying_instruction_cnt() > kHighWaterMark) {
      JUST(Global<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
        auto bc = std::make_shared<BlockingCounter>(1);
        engine_->InsertProbe([bc](vm::VirtualMachineEngine* engine) {
          const int64_t kLowWaterMark = GetInstructionLowWaterMark();
          if (engine->flying_instruction_cnt() > kLowWaterMark) { return false; }
          bc->Decrease();
          return true;
        });
        pending_notifier_.Notify();
        JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
        return Maybe<void>::Ok();
      }));
    }
    if (JUST(engine_->Receive(instruction_list))) {
      // old scheduler_pending_instruction_list is empty.
      pending_notifier_.Notify();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> VirtualMachine::NotifyOrRunScheduler() {
  if (unlikely(pthread_fork::IsForkedSubProcess() || disable_vm_threads_)) {
    ScheduleUntilVMEmpty(engine_.Mutable(), SingleThreadScheduleCtx());
  } else {
    pending_notifier_.Notify();
  }
  return Maybe<void>::Ok();
}

Maybe<void> VirtualMachine::RunInCurrentThread(vm::InstructionList* instr_list) {
  CHECK_OR_RETURN(engine_->SchedulerEmpty())
      << "vm scheduler not empty. May be a fatal error occured";
  JUST(engine_->Receive(instr_list));
  ScheduleUntilVMEmpty(engine_.Mutable(), SingleThreadScheduleCtx());
  return Maybe<void>::Ok();
}

namespace {

class MultiThreadScheduleCtx : public vm::ScheduleCtx {
 public:
  MultiThreadScheduleCtx() = default;
  ~MultiThreadScheduleCtx() = default;

  void OnWorkerLoadPending(vm::ThreadCtx* thread_ctx) const override {
    thread_ctx->mut_notifier()->Notify();
  }
};

}  // namespace

void VirtualMachine::ScheduleLoop(const std::function<void()>& Initializer) {
  Initializer();
  MultiThreadScheduleCtx schedule_ctx{};
  while (pending_notifier_.WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) {
    OF_PROFILER_RANGE_GUARD("VirtualMachine::ScheduleLoop");
    auto start = std::chrono::steady_clock::now();
    static constexpr int kWorkingMicroseconds = 1000;
    // Every time this thread wakes up, engine_ is scheduled for about `kWorkingMicroseconds`.
    // The cost of os thread switching is about 5-10 microseconds. Doing more scheduling in
    // a single waiting up can reach higher performance.
    do {
      static constexpr int kNumSchedulingPerTimoutTest = 10000;
      // Every time kWorkingMicroseconds timeout tested, engine_ is scheduled for about
      // kNumSchedulingPerTimoutTest.
      // The cost of `MicrosecondsFrom(start)` is about 400ns, while the empty scheduling costs
      // about 10ns.
      int i = 0;
      do {
        // Use SchedulerThreadUnsafeEmpty to avoid acquiring mutex lock.
        // It's safe to use SchedulerThreadUnsafeEmpty here. pending_notifier_.notified_cnt_ will be
        // greater than zero when inconsistency between
        // engine_->pending_instruction_list.list_head_.list_head_.container_ and
        // engine_->pending_instruction_list.list_head_.list_head_.size_ occured. hence the pending
        // instructions
        // will get handled in the next iteration.
        //  VirtualMachine::Receive may be less effiencient if the thread safe version
        //  `engine_->SchedulerEmpty()`
        // used
        //  here, because VirtualMachine::ScheduleLoop is more likely to get the mutex lock.
        do { engine_->Schedule(schedule_ctx); } while (!engine_->SchedulerThreadUnsafeEmpty());
      } while (++i < kNumSchedulingPerTimoutTest);
    } while (MicrosecondsFrom(start) < kWorkingMicroseconds);
  }
  ScheduleUntilVMEmpty(engine_.Mutable(), schedule_ctx);
  CHECK_JUST(ForEachThreadCtx(engine_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    thread_ctx->mut_notifier()->Close();
    return Maybe<void>::Ok();
  }));
  {
    std::unique_lock<std::mutex> lock(worker_threads_mutex_);
    for (const auto& worker_thread : worker_threads_) { worker_thread->join(); }
  }
  scheduler_stopped_ = true;
}

intrusive::shared_ptr<vm::MirroredObject> VirtualMachine::FindOrCreateScheduleLocalDepObject(
    Symbol<Device> device, StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  auto key = std::make_pair(device, stream_role);
  intrusive::shared_ptr<vm::MirroredObject>* ptr = &device_stream_role2local_dep_object_[key];
  if (!*ptr) { *ptr = intrusive::make_shared<vm::MirroredObject>(); }
  return *ptr;
}

intrusive::shared_ptr<vm::MirroredObject> VirtualMachine::FindOrCreateTransportLocalDepObject() {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  if (!transport_local_dep_object_) {
    transport_local_dep_object_ = intrusive::make_shared<vm::MirroredObject>();
  }
  return transport_local_dep_object_;
}

Maybe<vm::Stream*> VirtualMachine::CreateStream(Symbol<Device> device, StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  vm::ThreadCtx* thread_ctx = JUST(FindOrCreateThreadCtx(device, stream_role));
  return JUST(CreateStream(thread_ctx, device, stream_role));
}

Maybe<vm::Stream*> VirtualMachine::GetVmStream(Symbol<Stream> stream) {
  if (stream->unique_stream_id() >= unique_stream_id2vm_stream_.size()) {
    std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
    if (stream->unique_stream_id() >= unique_stream_id2vm_stream_.size()) {
      auto* stream_mgr = JUST(GlobalMaybe<StreamMgr>());
      for (int i = unique_stream_id2vm_stream_.size(); i <= stream->unique_stream_id(); ++i) {
        Symbol<Stream> cur_stream = JUST(stream_mgr->GetStreamSymbol(i));
        CHECK_EQ_OR_RETURN(cur_stream->unique_stream_id(), i)
            << "invalid Stream::unique_stream_id()";
        *unique_stream_id2vm_stream_.MutableOrAdd(cur_stream->unique_stream_id()) =
            JUST(CreateStream(cur_stream->device(), cur_stream->stream_role()));
      }
    }
  }
  return JUST(VectorAt(unique_stream_id2vm_stream_, stream->unique_stream_id()));
}

Maybe<vm::ThreadCtx*> VirtualMachine::FindOrCreateThreadCtx(Symbol<Device> device,
                                                            StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  vm::ThreadCtx** thread_ctx_ptr = nullptr;
  if (StreamOnIndependentThread::Visit(stream_role)) {
    auto key = std::make_pair(device->enum_type(), stream_role);
    thread_ctx_ptr = &devcie_type_stream_role_2independent_thread_ctx_[key];
  } else {
    thread_ctx_ptr = &devcie_type2non_independent_thread_ctx_[device->enum_type()];
  }
  if (*thread_ctx_ptr == nullptr) { *thread_ctx_ptr = JUST(CreateThreadCtx(device, stream_role)); }
  return *thread_ctx_ptr;
}

Maybe<vm::ThreadCtx*> VirtualMachine::CreateThreadCtx(Symbol<Device> device,
                                                      StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  // thread_ctx_ptr may be used after timout.
  auto thread_ctx_ptr = std::make_shared<vm::ThreadCtx*>(nullptr);
  {
    auto bc = std::make_shared<BlockingCounter>(1);
    engine_->InsertProbe([thread_ctx_ptr, bc](vm::VirtualMachineEngine* engine) {
      auto thread_ctx = intrusive::make_shared<vm::ThreadCtx>();
      engine->mut_thread_ctx_list()->PushBack(thread_ctx.Mutable());
      *thread_ctx_ptr = thread_ctx.Mutable();
      bc->Decrease();
      return true;
    });
    JUST(NotifyOrRunScheduler());
    JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  }
  auto* thread_ctx = *thread_ctx_ptr;
  {
    const auto& WorkerInitializer = [device, stream_role](vm::ThreadCtx* thread_ctx) {
      int device_type_value = static_cast<int>(device->enum_type());
      CHECK_GT(device_type_value, 0);
      std::string device_tag = *CHECK_JUST(DeviceTag4DeviceType(device->enum_type()));
      if (!StreamOnIndependentThread::Visit(stream_role)) {
        CHECK_JUST(InitThisThreadConsistentId(device_type_value + kThreadConsistentIdScheduler,
                                              device_tag));
      }
      OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Worker_" + device_tag);
    };
    auto thread = std::make_unique<std::thread>(&WorkerLoop, thread_ctx, WorkerInitializer);
    {
      std::unique_lock<std::mutex> lock(worker_threads_mutex_);
      worker_threads_.push_back(std::move(thread));
    }
  }
  return thread_ctx;
}

Maybe<vm::Stream*> VirtualMachine::CreateStream(vm::ThreadCtx* thread_ctx, Symbol<Device> device,
                                                StreamRole stream_role) {
  std::unique_lock<std::recursive_mutex> lock(creating_stream_and_thread_ctx_mutex_);
  // stream_ptr may be used after timout.
  auto stream_ptr = std::make_shared<vm::Stream*>(nullptr);
  auto bc = std::make_shared<BlockingCounter>(1);
  intrusive::shared_ptr<vm::MirroredObject> schedule_local_dep_object =
      FindOrCreateScheduleLocalDepObject(device, stream_role);
  Optional<intrusive::shared_ptr<vm::MirroredObject>> transport_local_dep_object;
  if (IsCommNetStream::Visit(stream_role)) {
    transport_local_dep_object = FindOrCreateTransportLocalDepObject();
  }
  engine_->InsertProbe([stream_ptr, thread_ctx, device, stream_role, bc, schedule_local_dep_object,
                        transport_local_dep_object](vm::VirtualMachineEngine* engine) {
    auto stream = intrusive::make_shared<vm::Stream>(
        thread_ctx, device, stream_role, schedule_local_dep_object, transport_local_dep_object);
    thread_ctx->mut_stream_list()->PushBack(stream.Mutable());
    *stream_ptr = stream.Mutable();
    bc->Decrease();
    return true;
  });
  JUST(NotifyOrRunScheduler());
  JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return *stream_ptr;
}

}  // namespace oneflow
