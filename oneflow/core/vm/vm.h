#ifndef ONEFLOW_CORE_VM_VM_H_
#define ONEFLOW_CORE_VM_VM_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vm_type.h"
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/vm/scheduler.msg.h"

namespace oneflow {

template<vm::VmType vm_type>
class OneflowVM final {
 public:
  OneflowVM(const Resource& resource, int64_t this_machine_id)
      : scheduler_(ObjectMsgPtr<vm::Scheduler>::New(
            vm::MakeVmDesc<vm_type>(resource, this_machine_id).Get())) {}
  ~OneflowVM() = default;

  vm::Scheduler* mut_scheduler() { return scheduler_.Mutable(); }

 private:
  ObjectMsgPtr<vm::Scheduler> scheduler_;
};

namespace vm {

class InstructionListProto;
class InstructionMsg;

ObjectMsgPtr<InstructionMsg> NewInstruction(const std::string& instr_type_name);

Maybe<void> Run(const InstructionListProto& instruction_list_proto);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_H_
