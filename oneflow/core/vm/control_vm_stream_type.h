#ifndef ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"

namespace oneflow {
namespace vm {

class Scheduler;
class InstructionMsg;

class ControlStreamType final : public StreamType {
 public:
  ControlStreamType() = default;
  ~ControlStreamType() = default;

  static const StreamTypeId kStreamTypeId = 0;

  ObjectMsgPtr<InstructionMsg> NewSymbol(const LogicalObjectId& logical_object_id,
                                         int64_t parallel_num) const;
  ObjectMsgPtr<InstructionMsg> DeleteSymbol(const LogicalObjectId& logical_object_id) const;

  bool IsSourceOpcode(InstructionOpcode opcode) const;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* vm_stream) const override {}

  void InitInstructionStatus(const Stream& vm_stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& vm_stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& vm_stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Run(InstrChain* vm_instr_chain) const override;

  void Run(Scheduler* scheduler, InstructionMsg* vm_instr_msg) const;
  void Run(Scheduler* scheduler, InstrChain* vm_instr_chain) const;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_
