#include "oneflow/core/vm/vpu_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VpuInstructionCtx::__Init__(VpuInstructionMsg* vpu_instruction_msg, VpuCtx* vpu_ctx) {
  reset_vpu_instruction_msg(vpu_instruction_msg);
  const auto& factory = vpu_ctx->vpu_set_ctx().vpu_type_ctx().vpu_instruction_factory();
  VpuInstructionOpcode opcode = vpu_instruction_msg->vpu_instruction_proto().opcode();
  set_vpu_instruction(factory.GetVpuInstruction(opcode));
  set_vpu_ctx(vpu_ctx);
}

void RunningVpuInstructionPackage::__Init__(VpuCtx* vpu_ctx) {
  set_vpu_ctx(vpu_ctx);
  const auto* factory = &vpu_ctx->vpu_set_ctx().vpu_type_ctx().vpu_instruction_factory();
  mutable_status_querier()->__Init__(
      [factory, vpu_ctx](ObjectMsgAllocator* allocator, int32_t* size) {
        return factory->NewStatusQuerier(allocator, size, vpu_ctx);
      });
}

}  // namespace oneflow
