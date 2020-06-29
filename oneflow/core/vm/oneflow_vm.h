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
  ~OneflowVM() = default;

  vm::VirtualMachine* mut_vm() { return vm_.Mutable(); }
  void TryReceiveAndRun();

 private:
  ObjectMsgPtr<vm::VirtualMachine> vm_;
  HashMap<vm::ThreadCtx*, std::unique_ptr<ThreadPool>> thread_ctx2thread_pool_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ONEFLOW_VM_H_
