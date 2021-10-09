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
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

class Resource;

namespace vm {

struct Stream;
struct Instruction;
struct InstructionStatusBuffer;

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

  intrusive::shared_ptr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                                   int64_t this_machine_id) const override {
    auto stream_desc = T().MakeStreamDesc(resource, this_machine_id);
    if (stream_desc) {
      stream_desc->mut_stream_type_id()->CopyFrom(
          LookupInferStreamTypeId(stream_desc->stream_type_id()));
    }
    return stream_desc;
  }
  bool SharingVirtualMachineThread() const override { return true; }
  bool SupportingTransportInstructions() const override { return false; }
};

template<>
class InferStreamType<ControlStreamType> final : public StreamType {
 public:
  InferStreamType() = default;
  ~InferStreamType() = default;

  const char* device_tag() const override { return "cpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override {
    return ControlStreamType().InitInstructionStatus(stream, status_buffer);
  }
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override {
    return ControlStreamType().DeleteInstructionStatus(stream, status_buffer);
  }
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override {
    return ControlStreamType().QueryInstructionStatusDone(stream, status_buffer);
  }
  void Infer(Instruction* instruction) const override { UNIMPLEMENTED(); }
  void Infer(VirtualMachine* vm, Instruction* instruction) const override {
    ControlStreamType().Infer(vm, instruction);
  }
  void Infer(VirtualMachine* vm, InstructionMsg* instruction_msg) const override {
    ControlStreamType().Infer(vm, instruction_msg);
  }
  void Compute(Instruction* instruction) const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void Compute(VirtualMachine*, InstructionMsg*) const override { LOG(FATAL) << "UNIMPLEMENTED"; }

  bool SharingVirtualMachineThread() const override { return true; }
  bool SupportingTransportInstructions() const override { return false; }
  bool IsControlStreamType() const override { return true; }

  intrusive::shared_ptr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                                   int64_t this_machine_id) const override {
    auto stream_desc = ControlStreamType().MakeStreamDesc(resource, this_machine_id);
    stream_desc->mut_stream_type_id()->CopyFrom(
        LookupInferStreamTypeId(stream_desc->stream_type_id()));
    return stream_desc;
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INFER_STREAM_TYPE_H_
