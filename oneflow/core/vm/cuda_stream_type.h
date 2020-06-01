#ifndef ONEFLOW_CORE_VM_CUDA_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_CUDA_STREAM_TYPE_H_

#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class CudaStreamType final : public StreamType {
 public:
  CudaStreamType() = default;
  ~CudaStreamType() override = default;

  const char* device_tag() const override { return "gpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override;

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Compute(Instruction* instruction) const override;
  ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                          int64_t this_machine_id) const override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_STREAM_TYPE_H_
