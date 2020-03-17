#ifndef ONEFLOW_CORE_VM_CUDA_COPY_D2H_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_CUDA_COPY_D2H_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/stream_type.h"

namespace oneflow {
namespace vm {

class Scheduler;
class InstructionMsg;

class CudaCopyD2HStreamType final : public StreamType {
 public:
  CudaCopyD2HStreamType() = default;
  ~CudaCopyD2HStreamType() = default;

  static const StreamTypeId kStreamTypeId = 5;

  ObjectMsgPtr<InstructionMsg> Copy(uint64_t dst, uint64_t src, size_t size) const;

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

#endif  // ONEFLOW_CORE_VM_CUDA_COPY_D2H_VM_STREAM_TYPE_H_
