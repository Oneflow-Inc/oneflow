#ifndef ONEFLOW_CORE_VM_CONTROL_VPU_H_
#define ONEFLOW_CORE_VM_CONTROL_VPU_H_

#include "oneflow/core/vm/vpu.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class InvalidVpu final : public Vpu {
 public:
  InvalidVpu() : Vpu() {}
  ~InvalidVpu() override = default;

  void Run(VpuSchedulerCtx* scheduler, VpuInstructionMsg* vpu_instr_msg) const;

  // UNIMPLEMENTED methods
  const VpuInstruction* GetVpuInstruction(VpuInstructionOpcode vpu_instr_opcode) const override {
    UNIMPLEMENTED();
    return nullptr;
  }
  VpuInstructionStatusQuerier* NewStatusQuerier(ObjectMsgAllocator* allocator, int* allocated_size,
                                                const VpuCtx* vpu_ctx) const override {
    UNIMPLEMENTED();
    return nullptr;
  }
  void Run(VpuCtx* vpu_ctx, RunningVpuInstructionPackage* vpu_instr_pkg) const override {
    UNIMPLEMENTED();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_VPU_H_
