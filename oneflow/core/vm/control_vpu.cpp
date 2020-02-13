#include "oneflow/core/vm/control_vpu.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void ControlVpu::Run(VpuSchedulerCtx* scheduler, VpuInstructionMsg* vpu_instr_msg) const { TODO(); }

const VpuInstruction* ControlVpu::GetVpuInstruction(VpuInstructionOpcode vpu_instr_opcode) const {
  UNIMPLEMENTED();
  return nullptr;
}

VpuInstructionStatusQuerier* ControlVpu::NewStatusQuerier(ObjectMsgAllocator* allocator,
                                                          int* allocated_size,
                                                          const VpuCtx* vpu_ctx) const {
  UNIMPLEMENTED();
  return nullptr;
}

void ControlVpu::Run(VpuCtx* vpu_ctx, RunningVpuInstructionPackage* vpu_instr_pkg) const {
  UNIMPLEMENTED();
}

}  // namespace oneflow
