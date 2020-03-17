#ifndef ONEFLOW_CORE_VM_L2R_RECEIVER_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_L2R_RECEIVER_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/vm_stream_type.h"

namespace oneflow {
namespace vm {

class Scheduler;
class InstructionMsg;

class L2RReceiverStreamType final : public StreamType {
 public:
  L2RReceiverStreamType() = default;
  ~L2RReceiverStreamType() = default;

  static const StreamTypeId kStreamTypeId = 7;

  ObjectMsgPtr<InstructionMsg> Receive(uint64_t logical_token, uint64_t dst, size_t size) const;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* vm_stream) const override;

  void InitInstructionStatus(const Stream& vm_stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& vm_stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& vm_stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Run(InstrChain* vm_instr_chain) const override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_L2R_RECEIVER_VM_STREAM_TYPE_H_
