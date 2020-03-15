#ifndef ONEFLOW_CORE_VM_HOST_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_HOST_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/vm_stream_type.h"

namespace oneflow {

class VmScheduler;
class VmInstructionMsg;

class HostVmStreamType final : public VmStreamType {
 public:
  HostVmStreamType() = default;
  ~HostVmStreamType() = default;

  static const VmStreamTypeId kVmStreamTypeId = 2;

  ObjectMsgPtr<VmInstructionMsg> Malloc(uint64_t logical_object_id, size_t size) const;
  ObjectMsgPtr<VmInstructionMsg> Free(uint64_t logical_object_id) const;

  ObjectMsgPtr<VmInstructionMsg> CudaMallocHost(uint64_t logical_object_id, size_t size) const;
  ObjectMsgPtr<VmInstructionMsg> CudaFreeHost(uint64_t logical_object_id) const;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, VmStream* vm_stream) const override {}

  void InitVmInstructionStatus(const VmStream& vm_stream,
                               VmInstructionStatusBuffer* status_buffer) const override;
  void DeleteVmInstructionStatus(const VmStream& vm_stream,
                                 VmInstructionStatusBuffer* status_buffer) const override;
  bool QueryVmInstructionStatusDone(const VmStream& vm_stream,
                                    const VmInstructionStatusBuffer& status_buffer) const override;
  void Run(VmInstrChain* vm_instr_chain) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_HOST_VM_STREAM_TYPE_H_
