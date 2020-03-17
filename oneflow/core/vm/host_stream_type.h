#ifndef ONEFLOW_CORE_VM_HOST_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_HOST_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/stream_type.h"

namespace oneflow {
namespace vm {

class Scheduler;
class InstructionMsg;

class HostStreamType final : public StreamType {
 public:
  HostStreamType() = default;
  ~HostStreamType() = default;

  static const StreamTypeId kStreamTypeId = 2;

  ObjectMsgPtr<InstructionMsg> Malloc(uint64_t logical_object_id, size_t size) const;
  ObjectMsgPtr<InstructionMsg> Free(uint64_t logical_object_id) const;

  ObjectMsgPtr<InstructionMsg> CudaMallocHost(uint64_t logical_object_id, size_t size) const;
  ObjectMsgPtr<InstructionMsg> CudaFreeHost(uint64_t logical_object_id) const;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Run(InstrChain* instr_chain) const override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_HOST_VM_STREAM_TYPE_H_
