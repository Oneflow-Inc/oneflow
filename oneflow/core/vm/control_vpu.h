#ifndef ONEFLOW_CORE_VM_CONTROL_VPU_H_
#define ONEFLOW_CORE_VM_CONTROL_VPU_H_

#include "oneflow/core/vm/vpu.h"
#include "oneflow/core/vm/vpu_instruction.msg.h"

namespace oneflow {

class VpuSchedulerCtx;
class VpuInstructionMsg;

class ControlVpu final {
 public:
  ControlVpu() = default;
  ~ControlVpu() = default;

  ObjectMsgPtr<VpuInstructionMsg> NewMirroredObjectSymbol(uint64_t symbol, bool is_remote,
                                                          int64_t parallel_num) const;
  ObjectMsgPtr<VpuInstructionMsg> DeleteMirroredObjectSymbol(
      const LogicalObjectId& logical_object_id) const;

  void Run(VpuSchedulerCtx* scheduler, VpuInstructionMsg* vpu_instr_msg) const;
};

static const VpuTypeId kControlVpuTypeId = 0;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_VPU_H_
