#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmInstructionCtx::__Init__(VmInstructionMsg* vm_instruction_msg, VmStream* vm_stram) {
  reset_vm_instruction_msg(vm_instruction_msg);
  const auto& vm_stream_type = vm_stram->vm_thread().vpu_type_ctx().vm_stream_type();
  VmInstructionOpcode opcode = vm_instruction_msg->vm_instruction_proto().opcode();
  set_vm_instruction(vm_stream_type.GetVmInstruction(opcode));
  set_vm_stram(vm_stram);
}

void RunningVmInstructionPackage::__Init__(VmStream* vm_stram) {
  set_vm_stram(vm_stram);
  const auto* vm_stream_type = &vm_stram->vm_thread().vpu_type_ctx().vm_stream_type();
  mutable_status_querier()->__Init__(
      [vm_stream_type, vm_stram](ObjectMsgAllocator* allocator, int32_t* size) {
        return vm_stream_type->NewStatusQuerier(allocator, size, vm_stram);
      });
}

}  // namespace oneflow
