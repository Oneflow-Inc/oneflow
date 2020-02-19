#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmInstructionCtx::__Init__(VmInstructionMsg* vm_instruction_msg, VmStream* vm_stram) {
  reset_vm_instruction_msg(vm_instruction_msg);
  const auto& vpu = vm_stram->vm_thread().vpu_type_ctx().vpu();
  VmInstructionOpcode opcode = vm_instruction_msg->vm_instruction_proto().opcode();
  set_vm_instruction(vpu.GetVmInstruction(opcode));
  set_vm_stram(vm_stram);
}

void RunningVmInstructionPackage::__Init__(VmStream* vm_stram) {
  set_vm_stram(vm_stram);
  const auto* vpu = &vm_stram->vm_thread().vpu_type_ctx().vpu();
  mutable_status_querier()->__Init__([vpu, vm_stram](ObjectMsgAllocator* allocator, int32_t* size) {
    return vpu->NewStatusQuerier(allocator, size, vm_stram);
  });
}

}  // namespace oneflow
