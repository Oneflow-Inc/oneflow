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

#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

namespace vm {

class ThreadCtx;
}

class OneflowVM final {
 public:
  OneflowVM(const OneflowVM&) = delete;
  OneflowVM(OneflowVM&&) = delete;
  OneflowVM(const Resource& resource, int64_t this_machine_id);
  ~OneflowVM();

  vm::VirtualMachine* mut_vm() { return vm_.Mutable(); }
  void TryReceiveAndRun();

 private:
  ObjectMsgPtr<vm::VirtualMachine> vm_;
  HashMap<vm::ThreadCtx*, std::unique_ptr<ThreadPool>> thread_ctx2thread_pool_;
  std::thread loop_thread_;
  bool exiting_;
  mutable std::mutex mutex_;

  void Loop();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ONEFLOW_VM_H_
