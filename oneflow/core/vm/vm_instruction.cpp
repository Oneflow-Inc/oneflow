#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmInstructionCtx::__Init__(VmInstructionMsg* vm_instruction_msg, VmStream* vm_stream) {
  reset_vm_instruction_msg(vm_instruction_msg);
  const auto& vm_stream_type = vm_stream->vm_thread().vm_stream_rt_desc().vm_stream_type();
  VmInstructionOpcode opcode = vm_instruction_msg->vm_instruction_proto().opcode();
  set_vm_instruction(vm_stream_type.GetVmInstruction(opcode));
  set_vm_stream(vm_stream);
}

void RunningVmInstructionPackage::__Init__(VmStream* vm_stream) {
  set_vm_stream(vm_stream);
  const auto* vm_stream_type = &vm_stream->vm_thread().vm_stream_rt_desc().vm_stream_type();
  mutable_status_querier()->__Init__(
      [vm_stream_type, vm_stream](ObjectMsgAllocator* allocator, int32_t* size) {
        return vm_stream_type->NewStatusQuerier(allocator, size, vm_stream);
      });
}

}  // namespace oneflow
