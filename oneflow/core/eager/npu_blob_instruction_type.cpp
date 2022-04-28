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
#ifdef WITH_NPU
#include "oneflow/core/vm/npu_stream_type.h"
#include "oneflow/core/eager/blob_instruction_type.h"
#include "oneflow/core/vm/npu_optional_event_record_status_querier.h"
#include "oneflow/core/vm/stream.h"
//#include "oneflow/core/vm/async_npu_stream_type.h"
#include "oneflow/core/device/npu_event.h"

namespace oneflow {
namespace vm {

class NpuAccessBlobByCallbackInstructionType final : public AccessBlobByCallbackInstructionType {
 public:
  NpuAccessBlobByCallbackInstructionType() = default;
  ~NpuAccessBlobByCallbackInstructionType() override = default;
  using stream_type = vm::NpuStreamType;
};
COMMAND(vm::RegisterInstructionType<NpuAccessBlobByCallbackInstructionType>(
    "npu.AccessBlobByCallback"));

class NpuTensorViewInstructionType final : public TensorViewInstructionType {
 public:
  NpuTensorViewInstructionType() = default;
  ~NpuTensorViewInstructionType() override = default;

  using stream_type = vm::NpuStreamType;
};
COMMAND(vm::RegisterInstructionType<NpuTensorViewInstructionType>("npu.TensorView"));

class NpuRecordEventInstructionType : public RecordEventInstructionType {
 public:
  NpuRecordEventInstructionType() = default;
  ~NpuRecordEventInstructionType() override = default;
  using stream_type = vm::NpuStreamType;

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAsTailOnly; }

  void InitInstructionStatus(Instruction* instruction) const override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_type().InitInstructionStatus(*stream, status_buffer);
    auto* event_provider = dynamic_cast<QueryNpuEventProvider*>(stream->device_ctx().get());
    const auto& npu_event = CHECK_NOTNULL(event_provider)->GetNpuEvent();
    auto* data_ptr = status_buffer->mut_buffer()->mut_data();
    NpuOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_npu_event(npu_event);
  }
};
COMMAND(vm::RegisterInstructionType<NpuRecordEventInstructionType>("npu.RecordEvent"));

}  // namespace vm
}  // namespace oneflow
#endif
