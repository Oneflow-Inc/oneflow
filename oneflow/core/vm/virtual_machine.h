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
#include "oneflow/core/common/stream_role.h"
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

 private:
  friend class InstructionsBuilder;

  void ScheduleLoop(const std::function<void()>& Initializer);

  intrusive::shared_ptr<vm::Dependence> FindOrCreateScheduleLocalDepObject(Symbol<Device> device,
                                                                           StreamRole stream_role);
  bool NoMoreErasedInstructions(size_t* last_total_erased_instruction_cnt) const;

  const vm::VirtualMachineEngine& engine() const { return *engine_; }
  vm::VirtualMachineEngine* mut_engine() { return engine_.Mutable(); }

  void ControlSync();
  Maybe<vm::ThreadCtx*> FindOrCreateThreadCtx(Symbol<Device> device, StreamRole stream_role);
  Maybe<vm::ThreadCtx*> CreateThreadCtx(Symbol<Device> device, StreamRole stream_role);
  Maybe<vm::Stream*> CreateStream(Symbol<Device> device, StreamRole stream_role);

  Maybe<vm::Stream*> CreateStream(vm::ThreadCtx* thread_ctx, Symbol<Device> device,
                                  StreamRole stream_role);

  Maybe<void> RunInCurrentThread(vm::InstructionList* instr_list);

  Maybe<void> BlockingRunProbeFunc(const std::function<bool(vm::VirtualMachineEngine*)>& prob_func);

  Maybe<void> NotifyOrRunScheduler();

  bool disable_vm_threads_;
  bool scheduler_stopped_;
  intrusive::shared_ptr<vm::VirtualMachineEngine> engine_;

  // for asynchronized execution
  std::mutex worker_threads_mutex_;
  std::list<std::unique_ptr<std::thread>> worker_threads_;

  // for creating vm::Stream and vm::ThreadCtx
  std::recursive_mutex creating_stream_and_thread_ctx_mutex_;
  HashMap<DeviceType, vm::ThreadCtx*> devcie_type2non_independent_thread_ctx_;
  HashMap<std::pair<DeviceType, StreamRole>, vm::ThreadCtx*>
      devcie_type_stream_role_2independent_thread_ctx_;
  HashMap<std::pair<Symbol<Device>, StreamRole>, intrusive::shared_ptr<vm::Dependence>>
      device_stream_role2local_dep_object_;
  intrusive::shared_ptr<vm::Dependence> transport_local_dep_object_;
  SteadyVector<vm::Stream*> unique_stream_id2vm_stream_;

  std::thread schedule_thread_;
  Notifier pending_notifier_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_
