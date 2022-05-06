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

  Maybe<vm::Stream*> CreateStream(Symbol<Device> device, StreamRole stream_role);
  vm::MirroredObject* FindOrCreateScheduleLocalDepObject(Symbol<Device> device,
                                                         StreamRole stream_role);
  vm::MirroredObject* FindOrCreateTransportLocalDepObject();

  bool NoMoreErasedInstructions(size_t* last_total_erased_instruction_cnt) const;
  std::string GetBlockingDebugString();

  Maybe<void> Receive(vm::InstructionMsgList* instr_list);

  const vm::VirtualMachineEngine& vm() const { return *vm_; }

  Maybe<void> CloseVMThreads();

 private:
  friend class InstructionsBuilder;

  void ScheduleLoop(const std::function<void()>& Initializer);
  void CallbackLoop(const std::function<void()>& Initializer);

  vm::VirtualMachineEngine* mut_vm() { return vm_.Mutable(); }
  void ControlSync();
  Maybe<vm::ThreadCtx*> FindOrCreateThreadCtx(Symbol<Device> device, StreamRole stream_role);
  Maybe<vm::ThreadCtx*> CreateThreadCtx(Symbol<Device> device, StreamRole stream_role);
  Maybe<vm::Stream*> CreateStream(vm::ThreadCtx* thread_ctx, Symbol<Device> device,
                                  StreamRole stream_role);

  Maybe<void> RunInCurrentThread(vm::InstructionMsgList* instr_list);

  Maybe<void> NotifyOrRunScheduler();

  bool disable_vm_threads_;
  bool scheduler_stopped_;
  intrusive::shared_ptr<vm::VirtualMachineEngine> vm_;

  // for asynchronized execution
  std::mutex worker_threads_mutex_;
  std::list<std::unique_ptr<std::thread>> worker_threads_;

  // for creating vm::Stream and vm::ThreadCtx
  std::recursive_mutex creating_stream_and_thread_ctx_mutex_;
  HashMap<DeviceType, vm::ThreadCtx*> devcie_type2non_independent_thread_ctx_;
  HashMap<std::pair<DeviceType, StreamRole>, vm::ThreadCtx*>
      devcie_type_stream_role_2independent_thread_ctx_;
  HashMap<std::pair<Symbol<Device>, StreamRole>, intrusive::shared_ptr<vm::MirroredObject>>
      device_stream_role2local_dep_object_;
  intrusive::shared_ptr<vm::MirroredObject> transport_local_dep_object_;

  std::thread schedule_thread_;
  Notifier pending_notifier_;
  std::thread callback_thread_;
  Notifier callback_notifier_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_
