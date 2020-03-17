#ifndef ONEFLOW_CORE_VM_NOP_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_NOP_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/stream_type.h"

namespace oneflow {
namespace vm {

class Scheduler;
class InstructionMsg;

class NopStreamType final : public StreamType {
 public:
  NopStreamType() = default;
  ~NopStreamType() = default;

  static const StreamTypeId kStreamTypeId = 1;

  ObjectMsgPtr<InstructionMsg> Nop() const;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Run(InstrChain* instr_chain) const override;
  ObjectMsgPtr<StreamDesc> MakeRemoteStreamDesc(const Resource& resource,
                                                int64_t this_machine_id) const override;
  ObjectMsgPtr<StreamDesc> MakeLocalStreamDesc(const Resource& resource) const override;
};
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_NOP_VM_STREAM_TYPE_H_
