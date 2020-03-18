#ifndef ONEFLOW_CORE_VM_ONEFLOW_VM_H_
#define ONEFLOW_CORE_VM_ONEFLOW_VM_H_

#include "oneflow/core/vm/vm_type.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/scheduler.msg.h"

namespace oneflow {

template<vm::VmType vm_type>
class OneflowVM final {
 public:
  OneflowVM(const OneflowVM&) = delete;
  OneflowVM(OneflowVM&&) = delete;
  OneflowVM(const Resource& resource, int64_t this_machine_id)
      : scheduler_(ObjectMsgPtr<vm::Scheduler>::New(
            vm::MakeVmDesc<vm_type>(resource, this_machine_id).Get())) {}
  ~OneflowVM() = default;

  vm::Scheduler* mut_scheduler() { return scheduler_.Mutable(); }

 private:
  ObjectMsgPtr<vm::Scheduler> scheduler_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ONEFLOW_VM_H_
