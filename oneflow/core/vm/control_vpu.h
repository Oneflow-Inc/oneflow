#ifndef ONEFLOW_CORE_VM_CONTROL_VPU_H_
#define ONEFLOW_CORE_VM_CONTROL_VPU_H_

#include "oneflow/core/vm/vpu.h"

namespace oneflow {

class VpuSchedulerCtx;
class VpuInstructionMsg;

class ControlVpu final : public Vpu {
 public:
  ControlVpu() : Vpu() {}
  ~ControlVpu() override = default;

  void Run(VpuSchedulerCtx* scheduler, VpuInstructionMsg* vpu_instr_msg) const;

  // UNIMPLEMENTED methods
  const VpuInstruction* GetVpuInstruction(VpuInstructionOpcode vpu_instr_opcode) const override;
  VpuInstructionStatusQuerier* NewStatusQuerier(ObjectMsgAllocator* allocator, int* allocated_size,
                                                const VpuCtx* vpu_ctx) const override;
  void Run(VpuCtx* vpu_ctx, RunningVpuInstructionPackage* vpu_instr_pkg) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_VPU_H_
