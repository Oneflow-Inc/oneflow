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
#include "oneflow/core/vm/cpu_stream_type.h"
#ifdef WITH_CUDA
#include "oneflow/core/eager/blob_instruction_type.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/cuda_optional_event_record_status_querier.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/async_cuda_stream_type.h"
#include "oneflow/core/device/cuda_event.h"

namespace oneflow {
namespace vm {

class GpuAccessBlobByCallbackInstructionType final : public AccessBlobByCallbackInstructionType {
 public:
  GpuAccessBlobByCallbackInstructionType() = default;
  ~GpuAccessBlobByCallbackInstructionType() override = default;
  using stream_type = vm::CudaStreamType;
};
COMMAND(vm::RegisterInstructionType<GpuAccessBlobByCallbackInstructionType>(
    "cuda.AccessBlobByCallback"));

class GpuRecordEventInstructionType : public RecordEventInstructionType {
 public:
  GpuRecordEventInstructionType() = default;
  ~GpuRecordEventInstructionType() override = default;
  using stream_type = vm::CudaStreamType;

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAsTailOnly; }

  void InitInstructionStatus(Instruction* instruction) const override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_type().InitInstructionStatus(*stream, status_buffer);
    auto* event_provider = dynamic_cast<QueryCudaEventProvider*>(stream->device_ctx().get());
    const auto& cuda_event = CHECK_NOTNULL(event_provider)->GetCudaEvent();
    auto* data_ptr = status_buffer->mut_buffer()->mut_data();
    CudaOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_cuda_event(cuda_event);
  }
};
COMMAND(vm::RegisterInstructionType<GpuRecordEventInstructionType>("cuda.RecordEvent"));

}  // namespace vm
}  // namespace oneflow
#endif
