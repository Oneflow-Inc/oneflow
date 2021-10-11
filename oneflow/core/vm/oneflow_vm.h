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
#ifndef ONEFLOW_CORE_VM_ONEFLOW_VM_H_
#define ONEFLOW_CORE_VM_ONEFLOW_VM_H_

#include "oneflow/core/common/notifier.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/prior_mutex.h"
#include "oneflow/core/thread/thread_consistent_id.h"
#include "oneflow/api/foreign_lock_helper.h"

namespace oneflow {

class OneflowVM final {
 public:
  OneflowVM(const OneflowVM&) = delete;
  OneflowVM(OneflowVM&&) = delete;
  OneflowVM(const Resource& resource, int64_t this_machine_id);
  ~OneflowVM();

  Maybe<void> Receive(vm::InstructionMsgList* instr_list);

  const vm::VirtualMachine& vm() const { return *vm_; }

 private:
  template<typename FinishConditionT>
  Maybe<void> ThisThreadScheduleUntil(vm::InstructionMsgList* instr_list,
                                      const FinishConditionT& FinishCondition) {
    const auto& Prepare = [&]() -> Maybe<void> { return mut_vm()->Receive(instr_list); };
    return HighPriorPrepareAndScheduleUntil(Prepare, FinishCondition);
  }
  template<typename FinishConditionT>
  Maybe<void> ThisThreadScheduleUntil(const FinishConditionT& FinishCondition) {
    return HighPriorPrepareAndScheduleUntil(&Maybe<void>::Ok, FinishCondition);
  }
  template<typename PrepareT, typename FinishConditionT>
  Maybe<void> HighPriorPrepareAndScheduleUntil(const PrepareT& Parpare,
                                               const FinishConditionT& FinishCondition) {
    // schedule vm until completion.
    {
      // make sure the thread_consistent_id be kThreadConsistentIdScheduler.
      // thread_consistent_id is essential for communicating with scheduler threads in other
      // processes.
      ThreadConsistentIdGurad guard(kThreadConsistentIdScheduler);
      HighPriorUniqueLock<PriorMutex> lock(prior_mutex_);
      JUST(Parpare());
      auto* vm = mut_vm();
      do { vm->Schedule(); } while (!JUST(FinishCondition()));
    }
    // It's quite likely that there are instructions left unscheduled even if FinishCondition() ==
    // true. Notify other threads do the remainder scheduling.
    notifier_.Notify();
    return Maybe<void>::Ok();
  }

  void Loop(const std::function<void()>& Initializer);

  vm::VirtualMachine* mut_vm() { return vm_.Mutable(); }
  void ControlSync();

  ObjectMsgPtr<vm::VirtualMachine> vm_;
  // for asynchronized execution
  std::list<std::unique_ptr<std::thread>> worker_threads_;
  std::thread schedule_thread_;
  std::atomic<bool> exiting_;
  Notifier notifier_;
  PriorMutex prior_mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ONEFLOW_VM_H_
