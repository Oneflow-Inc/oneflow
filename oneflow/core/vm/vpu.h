#ifndef ONEFLOW_CORE_VM_VPU_H_
#define ONEFLOW_CORE_VM_VPU_H_

#include "oneflow/core/vm/vpu_type_desc.msg.h"

namespace oneflow {

class VmInstruction {
 public:
  virtual ~VmInstruction() = default;

  virtual void Execute() const = 0;

 protected:
  VmInstruction() = default;
};

class VpuCtx;
class ObjectMsgAllocator;
class VmInstructionStatusQuerier;
class RunningVmInstructionPackage;

class Vpu {
 public:
  virtual ~Vpu() = default;

  virtual const VmInstruction* GetVmInstruction(VmInstructionOpcode vm_instr_opcode) const = 0;
  virtual VmInstructionStatusQuerier* NewStatusQuerier(ObjectMsgAllocator* allocator,
                                                       int* allocated_size,
                                                       const VpuCtx* vpu_ctx) const = 0;
  virtual void Run(VpuCtx* vpu_ctx, RunningVmInstructionPackage* vm_instr_pkg) const = 0;

 protected:
  Vpu() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_H_
