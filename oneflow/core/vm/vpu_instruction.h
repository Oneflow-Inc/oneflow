#ifndef ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_
#define ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_

#include "oneflow/core/vm/vpu_type_desc.msg.h"

namespace oneflow {

class VpuInstruction {
 public:
  virtual ~VpuInstruction() = default;

  virtual void Execute() const = 0;

 protected:
  VpuInstruction() = default;
};

class VpuCtx;
class ObjectMsgAllocator;
class VpuInstructionStatusQuerier;
class RunningVpuInstructionPackage;

class Vpu {
 public:
  virtual ~Vpu() = default;

  virtual const VpuInstruction* GetVpuInstruction(VpuInstructionOpcode vpu_instr_opcode) const = 0;
  virtual VpuInstructionStatusQuerier* NewStatusQuerier(ObjectMsgAllocator* allocator,
                                                        int* allocated_size,
                                                        const VpuCtx* vpu_ctx) const = 0;

  virtual void Run(VpuCtx* vpu_ctx, RunningVpuInstructionPackage* vpu_instr_pkg) const = 0;

 protected:
  Vpu() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_INSTRUCTION_H_
