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
#include <thread>
#include <chrono>
#include "oneflow/core/vm/sync_vm_mode_guard.h"
#include "oneflow/core/vm/barrier_instruction_policy.h"
#include "oneflow/core/vm/caching_allocator.h"
#include "oneflow/core/vm/global_sync_instruction_policy.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/thread/thread_global_id.h"
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/stream_on_independent_thread.h"
#include "oneflow/core/framework/stream_is_comm_net_stream.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/common/env_var/vm.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/stream_get_stream_type_name.h"
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
  *Initializer = [&]() { OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Scheduler"); };
}

void WorkerLoop(vm::ThreadCtx* thread_ctx, const std::function<void(vm::ThreadCtx*)>& Initializer) {
  SyncVmModeGuard guard(SyncVmMode::kEnable);
  Initializer(thread_ctx);
  constexpr static size_t kExpireMicroseconds = 200;
  while (thread_ctx->mut_notifier()->WaitAndClearNotifiedCnt() == kNotifierStatusSuccess) {
    std::chrono::time_point<std::chrono::steady_clock> start{};
    do {
      while (thread_ctx->TryReceiveAndRun()) { start = std::chrono::steady_clock::now(); }
      std::this_thread::yield();
    } while (MicrosecondsFrom(start) < kExpireMicroseconds);
  }
}

}  // namespace

VirtualMachine::VirtualMachine()
    : multi_thread_(ThreadLocalEnvBool<ONEFLOW_VM_MULTI_THREAD>()),
      threads_closed_(false),
      scheduler_stopped_(false) {
  // Class VirtualMachineEngine only cares the basic logical of vm, while class VirtualMachine
  // manages threads and condition variables.
  // In order to notify threads in VirtualMachineEngine, a notify callback lambda should be take as
  // an argument for VirtualMachineEngine's constructor.
  engine_ = intrusive::make_shared<vm::VirtualMachineEngine>();
  OF_PROFILER_NAME_THIS_HOST_THREAD("_Main");

  if (multi_thread_) {
    std::function<void()> SchedulerInitializer;
    GetSchedulerThreadInitializer(&SchedulerInitializer);
    schedule_thread_ = std::thread(&VirtualMachine::ScheduleLoop, this, SchedulerInitializer);
  }
  transport_dependence_.Reset();
}

namespace {

Maybe<Symbol<Stream>> GetBarrierStream() {
  auto device = JUST(Device::New("cpu"));
  return Stream::New(device, StreamType::kBarrier);
}

void MakeBarrierInstructions(vm::InstructionList* list,
                             const std::function<void()>& BarrierCallback) {
  auto* vm = Singleton<VirtualMachine>::Get();
  {
    auto stream = CHECK_JUST(GetBarrierStream());
    auto instruction = intrusive::make_shared<vm::Instruction>(
        CHECK_JUST(vm->GetVmStream(stream)), std::make_shared<vm::GlobalSyncInstructionPolicy>());
    list->EmplaceBack(std::move(instruction));
  }
  {
    auto stream = CHECK_JUST(GetBarrierStream());
    auto instruction = intrusive::make_shared<vm::Instruction>(
        CHECK_JUST(vm->GetVmStream(stream)),
        std::make_shared<vm::BarrierInstructionPolicy>(BarrierCallback));
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
  CHECK_OR_RETURN(!threads_closed_) << "vm threads closed";
  ControlSync();
  pending_notifier_.Close();
  if (multi_thread_) {
    schedule_thread_.join();
  } else {
    // For technical reasons, worker threads are always created even in single thread mode
    JUST(CloseWorkerThreads());
  }
  threads_closed_ = true;
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
  JUST(Singleton<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
    auto bc = std::make_shared<BlockingCounter>(1);
    engine_->InsertProbe([bc, prob_func](vm::VirtualMachineEngine* engine) {
      if (!prob_func(engine)) { return false; }
      bc->Decrease();
      return true;
    });
    if (threads_closed_ || !multi_thread_) {
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
        vm::Allocator* allocator = stream->mut_stream_policy()->mut_allocator();
        if (allocator) {
          auto* cache = dynamic_cast<vm::CachingAllocator*>(allocator);
          if (cache != nullptr) { cache->Shrink(); }
        }
      }
    }
    return true;
  };
  return BlockingRunProbeFunc(try_shrink_men);
}

VirtualMachine::~VirtualMachine() {
  if (!threads_closed_) { CHECK_JUST(CloseVMThreads()); }
  RunMainThreadPendingTasks();
  CHECK(engine_->SchedulerEmpty());
  engine_.Reset();
}

std::function<Maybe<bool>()> VirtualMachine::GetPredicatorNoMoreInstructionsFinished() {
  auto last_total_erased = std::make_shared<size_t>(0);
  auto* vm = Singleton<VirtualMachine>::Get();
  if (vm != nullptr) { *last_total_erased = vm->engine_->total_erased_instruction_cnt(); }
  return [last_total_erased]() -> Maybe<bool> {
    auto* vm = Singleton<VirtualMachine>::Get();
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

void VirtualMachine::RunMainThreadPendingTasks() {
  std::unique_lock lock(main_thread_pending_tasks_mutex_);
  for (const auto& main_thread_pending_task : main_thread_pending_tasks_) {
    main_thread_pending_task();
  }
  main_thread_pending_tasks_.clear();
}

Maybe<void> VirtualMachine::Receive(vm::InstructionList* instruction_list) {
  SyncVmModeGuard guard(SyncVmMode::kEnable);
  RunMainThreadPendingTasks();
  if (unlikely(pthread_fork::IsForkedSubProcess())) {
    INTRUSIVE_FOR_EACH_PTR(instruction, instruction_list) {
      const auto& device = instruction->stream().device();
      CHECK_OR_RETURN(device->enum_type() == DeviceType::kCPU)
          << pthread_fork::kOfCudaNotSupportInForkedSubProcess;
      JUST(instruction->Prepare());
      instruction->Compute();
    }
    instruction_list->Clear();
  } else if (unlikely(threads_closed_ || !multi_thread_)) {
    JUST(RunInCurrentThread(instruction_list));
  } else {
    const int64_t kHighWaterMark = GetInstructionHighWaterMark();
    if (engine_->flying_instruction_cnt() > kHighWaterMark) {
      JUST(Singleton<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
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
  if (unlikely(pthread_fork::IsForkedSubProcess() || threads_closed_ || !multi_thread_)) {
    ScheduleUntilVMEmpty(engine_.Mutable(), SingleThreadScheduleCtx());
  } else {
    pending_notifier_.Notify();
  }
  return Maybe<void>::Ok();
}

Maybe<void> VirtualMachine::CloseWorkerThreads() {
  JUST(ForEachThreadCtx(engine_.Mutable(), [&](vm::ThreadCtx* thread_ctx) -> Maybe<void> {
    thread_ctx->mut_notifier()->Close();
    return Maybe<void>::Ok();
  }));
  {
    std::unique_lock<std::mutex> lock(worker_threads_mutex_);
    for (const auto& worker_thread : worker_threads_) { worker_thread->join(); }
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
  SyncVmModeGuard guard(SyncVmMode::kEnable);
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
      do {
        const size_t total_inserted = engine_->total_inserted_instruction_cnt();
        const size_t total_erased = engine_->total_erased_instruction_cnt();
        engine_->Schedule(schedule_ctx);
        if (ThreadLocalEnvBool<ONEFLOW_VM_ENABLE_SCHEDULE_YIELD>()
            && total_inserted == engine_->total_inserted_instruction_cnt()
            && total_erased == engine_->total_erased_instruction_cnt()) {  // nothing handled.
          std::this_thread::yield();
        }
      } while (!engine_->SchedulerThreadUnsafeEmpty());
    } while (MicrosecondsFrom(start) < kWorkingMicroseconds);
  }
  ScheduleUntilVMEmpty(engine_.Mutable(), schedule_ctx);
  CHECK_JUST(CloseWorkerThreads());
  scheduler_stopped_ = true;
}

intrusive::shared_ptr<vm::Dependence> VirtualMachine::FindOrCreateScheduleDependence(
    Symbol<Stream> stream) {
  std::unique_lock<std::recursive_mutex> lock(stream_and_thread_ctx_mutex_);
  intrusive::shared_ptr<vm::Dependence>* ptr = &stream2dependence_[stream];
  if (!*ptr) { *ptr = intrusive::make_shared<vm::Dependence>(); }
  return *ptr;
}

intrusive::shared_ptr<vm::Dependence> VirtualMachine::FindOrCreateTransportLocalDepObject() {
  std::unique_lock<std::recursive_mutex> lock(stream_and_thread_ctx_mutex_);
  if (!transport_dependence_) { transport_dependence_ = intrusive::make_shared<vm::Dependence>(); }
  return transport_dependence_;
}

Maybe<vm::Stream*> VirtualMachine::CreateStream(Symbol<Stream> stream) {
  std::unique_lock<std::recursive_mutex> lock(stream_and_thread_ctx_mutex_);
  vm::ThreadCtx* thread_ctx =
      JUST(FindOrCreateThreadCtx(stream->device(), stream->stream_type(), stream->thread_uid()));
  return JUST(CreateStream(thread_ctx, stream));
}

Maybe<vm::Stream*> VirtualMachine::GetVmStream(Symbol<Stream> stream) {
  if (stream->unique_stream_id() >= unique_stream_id2vm_stream_.size()) {
    std::unique_lock<std::recursive_mutex> lock(stream_and_thread_ctx_mutex_);
    if (stream->unique_stream_id() >= unique_stream_id2vm_stream_.size()) {
      auto* stream_mgr = JUST(SingletonMaybe<StreamMgr>());
      for (int i = unique_stream_id2vm_stream_.size(); i <= stream->unique_stream_id(); ++i) {
        Symbol<Stream> cur_stream = JUST(stream_mgr->GetStreamSymbol(i));
        CHECK_EQ_OR_RETURN(cur_stream->unique_stream_id(), i)
            << "invalid Stream::unique_stream_id()";
        unique_stream_id2vm_stream_.SetOrAdd(cur_stream->unique_stream_id(),
                                             JUST(CreateStream(cur_stream)));
      }
    }
  }
  return JUST(VectorAt(unique_stream_id2vm_stream_, stream->unique_stream_id()));
}

Maybe<vm::ThreadCtx*> VirtualMachine::FindOrCreateThreadCtx(Symbol<Device> device,
                                                            StreamType stream_type,
                                                            size_t thread_uid) {
  std::unique_lock<std::recursive_mutex> lock(stream_and_thread_ctx_mutex_);
  vm::ThreadCtx** thread_ctx_ptr = nullptr;
  if (StreamOnIndependentThread::Visit(stream_type)) {
    auto key = std::make_pair(device->enum_type(), stream_type);
    thread_ctx_ptr = &devcie_type_stream_type_2independent_thread_ctx_[key];
  } else {
    thread_ctx_ptr = &thread_uid2shared_thread_ctx_[thread_uid];
  }
  if (*thread_ctx_ptr == nullptr) {
    *thread_ctx_ptr = JUST(CreateThreadCtx(device, stream_type, thread_uid));
  }
  return *thread_ctx_ptr;
}

Maybe<vm::ThreadCtx*> VirtualMachine::CreateThreadCtx(Symbol<Device> device, StreamType stream_type,
                                                      size_t thread_uid) {
  std::unique_lock<std::recursive_mutex> lock(stream_and_thread_ctx_mutex_);
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
    const std::string thread_tag = [&] {
      std::string device_tag = *CHECK_JUST(DeviceTag4DeviceType(device->enum_type()));
      if (StreamOnIndependentThread::Visit(stream_type)) {
        return device_tag + GetStreamTypeName::Visit(stream_type);
      } else {
        return std::to_string(thread_uid);
      }
    }();
    const auto& WorkerInitializer = [thread_tag](vm::ThreadCtx* thread_ctx) {
      OF_PROFILER_NAME_THIS_HOST_THREAD("_VM::Worker_" + thread_tag);
    };
    auto thread = std::make_unique<std::thread>(&WorkerLoop, thread_ctx, WorkerInitializer);
    {
      std::unique_lock<std::mutex> lock(worker_threads_mutex_);
      worker_threads_.push_back(std::move(thread));
    }
  }
  return thread_ctx;
}

Maybe<vm::Stream*> VirtualMachine::CreateStream(vm::ThreadCtx* thread_ctx, Symbol<Stream> stream) {
  std::unique_lock<std::recursive_mutex> lock(stream_and_thread_ctx_mutex_);
  intrusive::shared_ptr<vm::Dependence> schedule_dependence =
      FindOrCreateScheduleDependence(stream);
  std::vector<intrusive::shared_ptr<vm::Dependence>> transport_dependences{};
  if (IsCommNetStream::Visit(stream->stream_type())) {
    transport_dependences.push_back(FindOrCreateTransportLocalDepObject());
  }
  auto vm_stream =
      intrusive::make_shared<vm::Stream>(thread_ctx, stream->device(), stream->stream_type(),
                                         schedule_dependence, transport_dependences);

  auto bc = std::make_shared<BlockingCounter>(1);
  engine_->InsertProbe([&vm_stream, thread_ctx, bc](vm::VirtualMachineEngine* engine) {
    thread_ctx->mut_stream_list()->PushBack(vm_stream.Mutable());
    bc->Decrease();
    return true;
  });
  JUST(NotifyOrRunScheduler());
  JUST(bc->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return vm_stream.Mutable();
}

}  // namespace oneflow
