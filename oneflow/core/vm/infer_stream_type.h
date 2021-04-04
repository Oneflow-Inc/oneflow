/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_

#include <glog/logging.h>
#include "oneflow/core/object_msg/object_msg.h"
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
  bool SharingVirtualMachineThread() const override { return true; }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_
