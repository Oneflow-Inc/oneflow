#ifndef ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"

namespace oneflow {

class VmScheduler;
class VmInstructionMsg;

class ControlVmStreamType final : public VmStreamType {
 public:
  ControlVmStreamType() = default;
  ~ControlVmStreamType() = default;

  static const VmStreamTypeId kVmStreamTypeId = 0;

  ObjectMsgPtr<VmInstructionMsg> NewMirroredObjectSymbol(const LogicalObjectId& logical_object_id,
                                                         int64_t parallel_num) const;
  ObjectMsgPtr<VmInstructionMsg> DeleteMirroredObjectSymbol(
      const LogicalObjectId& logical_object_id) const;

  bool IsSourceOpcode(VmInstructionOpcode opcode) const;

  void InitVmInstructionStatus(const VmStream& vm_stream,
                               VmInstructionStatusBuffer* status_buffer) const override;
  void DeleteVmInstructionStatus(const VmStream& vm_stream,
                                 VmInstructionStatusBuffer* status_buffer) const override;
  bool QueryVmInstructionStatusDone(const VmStream& vm_stream,
                                    const VmInstructionStatusBuffer& status_buffer) const override;
  void Run(VmInstrChain* vm_instr_chain) const override;

  void Run(VmScheduler* scheduler, VmInstructionMsg* vm_instr_msg) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_
