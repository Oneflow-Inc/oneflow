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

#include "oneflow/core/common/notifier.h"
#include "oneflow/core/vm/vm_desc.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

class InstructionsBuilder;

class VirtualMachine final {
 public:
  VirtualMachine(const VirtualMachine&) = delete;
  VirtualMachine(VirtualMachine&&) = delete;
  VirtualMachine(const Resource& resource, int64_t this_machine_id);
  ~VirtualMachine();

  static std::function<Maybe<bool>()> GetPredicatorNoMoreInstructionsFinished();

  bool NoMoreErasedLivelyInstructions(size_t* last_total_erased_lively_instruction_cnt) const;
  std::string GetBlockingDebugString();

  Maybe<void> Receive(vm::InstructionMsgList* instr_list);

  const vm::VirtualMachineEngine& vm() const { return *vm_; }

 private:
  friend class InstructionsBuilder;

  void ScheduleLoop(const std::function<void()>& Initializer);
  void CallbackLoop(const std::function<void()>& Initializer);

  vm::VirtualMachineEngine* mut_vm() { return vm_.Mutable(); }
  void ControlSync();

  intrusive::shared_ptr<vm::VirtualMachineEngine> vm_;
  // for asynchronized execution
  std::list<std::unique_ptr<std::thread>> worker_threads_;
  std::thread schedule_thread_;
  Notifier pending_notifier_;
  std::thread callback_thread_;
  Notifier callback_notifier_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_
