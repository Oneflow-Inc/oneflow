#ifndef ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_

#include <glog/logging.h>
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

class Resource;

namespace vm {

class Stream;
class Instruction;
class InstructionStatusBuffer;

struct InferStreamTypeUtil final {
  static void InitInstructionStatus(const Stream& stream, InstructionStatusBuffer* status_buffer);
  static void DeleteInstructionStatus(const Stream& stream, InstructionStatusBuffer* status_buffer);
  static bool QueryInstructionStatusDone(const Stream& stream,
                                         const InstructionStatusBuffer& status_buffer);
  static void Infer(Instruction* instruction);
};

template<typename T>
class InferStreamType final : public StreamType {
 public:
  InferStreamType() = default;
  ~InferStreamType() = default;

  const char* device_tag() const override { return "cpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override {
    return InferStreamTypeUtil::InitInstructionStatus(stream, status_buffer);
  }
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override {
    return InferStreamTypeUtil::DeleteInstructionStatus(stream, status_buffer);
  }
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override {
    return InferStreamTypeUtil::QueryInstructionStatusDone(stream, status_buffer);
  }
  void Infer(Instruction* instruction) const override { InferStreamTypeUtil::Infer(instruction); }
  void Compute(Instruction* instruction) const override { LOG(FATAL) << "UNIMPLEMENTED"; }

  ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                          int64_t this_machine_id) const override {
    auto stream_desc = T().MakeStreamDesc(resource, this_machine_id);
    if (stream_desc) {
      stream_desc->mut_stream_type_id()->CopyFrom(
          LookupInferStreamTypeId(stream_desc->stream_type_id()));
    }
    return stream_desc;
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_
