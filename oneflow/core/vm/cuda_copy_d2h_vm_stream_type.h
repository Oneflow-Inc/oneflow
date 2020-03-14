#ifndef ONEFLOW_CORE_VM_CUDA_COPY_D2H_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_CUDA_COPY_D2H_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/vm_stream_type.h"

namespace oneflow {

class VmScheduler;
class VmInstructionMsg;

class CudaCopyD2HVmStreamType final : public VmStreamType {
 public:
  CudaCopyD2HVmStreamType() = default;
  ~CudaCopyD2HVmStreamType() = default;

  static const VmStreamTypeId kVmStreamTypeId = 5;

  ObjectMsgPtr<VmInstructionMsg> Copy(uint64_t dst, uint64_t src, size_t size) const;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, VmStream* vm_stream) const override;

  void InitVmInstructionStatus(const VmStream& vm_stream,
                               VmInstructionStatusBuffer* status_buffer) const override;
  void DeleteVmInstructionStatus(const VmStream& vm_stream,
                                 VmInstructionStatusBuffer* status_buffer) const override;
  bool QueryVmInstructionStatusDone(const VmStream& vm_stream,
                                    const VmInstructionStatusBuffer& status_buffer) const override;
  void Run(VmInstrChain* vm_instr_chain) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_COPY_D2H_VM_STREAM_TYPE_H_
