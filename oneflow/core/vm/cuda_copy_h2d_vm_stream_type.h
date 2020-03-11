#ifndef ONEFLOW_CORE_VM_CUDA_COPY_H2D_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_CUDA_COPY_H2D_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/vm_stream_type.h"

namespace oneflow {

class VmScheduler;
class VmInstructionMsg;

class CudaCopyH2DVmStreamType final : public VmStreamType {
 public:
  CudaCopyH2DVmStreamType() = default;
  ~CudaCopyH2DVmStreamType() = default;

  static const VmStreamTypeId kVmStreamTypeId = 4;

  ObjectMsgPtr<VmInstructionMsg> Copy(uint64_t dst, uint64_t src, size_t size) const;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                     CallbackMsgListPtr callback_list) const override;

  void InitVmInstructionStatus(const VmStream& vm_stream,
                               VmInstructionStatusBuffer* status_buffer) const override;
  void DeleteVmInstructionStatus(const VmStream& vm_stream,
                                 VmInstructionStatusBuffer* status_buffer) const override;
  bool QueryVmInstructionStatusDone(const VmStream& vm_stream,
                                    const VmInstructionStatusBuffer& status_buffer) const override;
  void Run(VmInstrChain* vm_instr_chain) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_COPY_H2D_VM_STREAM_TYPE_H_
