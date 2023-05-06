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
#ifndef ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_
#define ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_

#include <mutex>
#include "oneflow/core/common/notifier.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/common/steady_vector.h"

namespace oneflow {

class InstructionsBuilder;
class Device;

class VirtualMachine final {
 public:
  VirtualMachine(const VirtualMachine&) = delete;
  VirtualMachine(VirtualMachine&&) = delete;
  VirtualMachine();
  ~VirtualMachine();

  static std::function<Maybe<bool>()> GetPredicatorNoMoreInstructionsFinished();

  intrusive::shared_ptr<vm::Dependence> FindOrCreateTransportLocalDepObject();

  std::string GetBlockingDebugString();

  Maybe<void> Receive(vm::InstructionList* instr_list);

  Maybe<void> CloseVMThreads();

  // Never called in vm work threads.
  // VM sync must be called to ensure all working instructions are finished.
  Maybe<void> ShrinkAllMem();
  Maybe<vm::Stream*> GetVmStream(Symbol<Stream> stream);

  size_t flying_instruction_cnt() const { return engine().flying_instruction_cnt(); }

  void add_main_thread_pending_task(std::function<void()> task) {
    std::unique_lock lock(main_thread_pending_tasks_mutex_);
    main_thread_pending_tasks_.push_back(std::move(task));
  }

 private:
  friend class InstructionsBuilder;

  void ScheduleLoop(const std::function<void()>& Initializer);

  intrusive::shared_ptr<vm::Dependence> FindOrCreateScheduleDependence(Symbol<Stream> stream);
  bool NoMoreErasedInstructions(size_t* last_total_erased_instruction_cnt) const;

  const vm::VirtualMachineEngine& engine() const { return *engine_; }
  vm::VirtualMachineEngine* mut_engine() { return engine_.Mutable(); }

  void ControlSync();
  Maybe<vm::ThreadCtx*> FindOrCreateThreadCtx(Symbol<Device> device, StreamType stream_type,
                                              size_t thread_uid);
  Maybe<vm::ThreadCtx*> CreateThreadCtx(Symbol<Device> device, StreamType stream_type,
                                        size_t thread_uid);
  Maybe<vm::Stream*> CreateStream(Symbol<Stream> stream);

  Maybe<vm::Stream*> CreateStream(vm::ThreadCtx* thread_ctx, Symbol<Stream> stream);

  Maybe<void> RunInCurrentThread(vm::InstructionList* instr_list);

  Maybe<void> BlockingRunProbeFunc(const std::function<bool(vm::VirtualMachineEngine*)>& prob_func);

  Maybe<void> NotifyOrRunScheduler();

  Maybe<void> CloseWorkerThreads();

  void RunMainThreadPendingTasks();

  bool multi_thread_;
  bool threads_closed_;
  bool scheduler_stopped_;
  intrusive::shared_ptr<vm::VirtualMachineEngine> engine_;

  // for asynchronized execution
  std::mutex worker_threads_mutex_;
  std::list<std::unique_ptr<std::thread>> worker_threads_;

  // for vm::Stream and vm::ThreadCtx
  std::recursive_mutex stream_and_thread_ctx_mutex_;
  HashMap<size_t, vm::ThreadCtx*> thread_uid2shared_thread_ctx_;
  HashMap<std::pair<DeviceType, StreamType>, vm::ThreadCtx*>
      devcie_type_stream_type_2independent_thread_ctx_;
  HashMap<Symbol<Stream>, intrusive::shared_ptr<vm::Dependence>> stream2dependence_;
  intrusive::shared_ptr<vm::Dependence> transport_dependence_;
  SteadyVector<vm::Stream*> unique_stream_id2vm_stream_;

  std::thread schedule_thread_;
  Notifier pending_notifier_;

  std::mutex main_thread_pending_tasks_mutex_;
  std::vector<std::function<void()>> main_thread_pending_tasks_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_
