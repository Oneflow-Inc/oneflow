#ifndef ONEFLOW_CORE_VM_CONTROL_VPU_H_
#define ONEFLOW_CORE_VM_CONTROL_VPU_H_

#include "oneflow/core/vm/vpu.h"

namespace oneflow {

class VpuSchedulerCtx;
class VpuInstructionMsg;

class ControlVpu final {
 public:
  ControlVpu() = default;
  ~ControlVpu() = default;

  void Run(VpuSchedulerCtx* scheduler, VpuInstructionMsg* vpu_instr_msg) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_VPU_H_
