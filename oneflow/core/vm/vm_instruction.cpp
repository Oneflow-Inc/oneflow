#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmInstructionCtx::__Init__(VmInstructionMsg* vm_instruction_msg, VpuCtx* vpu_ctx) {
  reset_vm_instruction_msg(vm_instruction_msg);
  const auto& vpu = vpu_ctx->vpu_set_ctx().vpu_type_ctx().vpu();
  VmInstructionOpcode opcode = vm_instruction_msg->vm_instruction_proto().opcode();
  set_vm_instruction(vpu.GetVmInstruction(opcode));
  set_vpu_ctx(vpu_ctx);
}

void RunningVmInstructionPackage::__Init__(VpuCtx* vpu_ctx) {
  set_vpu_ctx(vpu_ctx);
  const auto* vpu = &vpu_ctx->vpu_set_ctx().vpu_type_ctx().vpu();
  mutable_status_querier()->__Init__([vpu, vpu_ctx](ObjectMsgAllocator* allocator, int32_t* size) {
    return vpu->NewStatusQuerier(allocator, size, vpu_ctx);
  });
}

}  // namespace oneflow
