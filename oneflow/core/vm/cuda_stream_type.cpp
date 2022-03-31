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
#ifdef WITH_CUDA

#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/cuda_optional_event_record_status_querier.h"
#include "oneflow/core/vm/cuda_stream_handle_device_context.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

void CudaStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const {
  device_ctx->reset(new CudaStreamHandleDeviceCtx(stream->device_id()));
}

void CudaStreamType::InitInstructionStatus(const Stream& stream,
                                           InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  auto* data_ptr = status_buffer->mut_buffer()->mut_data();
  CudaOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, nullptr);
}

void CudaStreamType::DeleteInstructionStatus(const Stream& stream,
                                             InstructionStatusBuffer* status_buffer) const {
  auto* ptr =
      CudaOptionalEventRecordStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data());
  ptr->~CudaOptionalEventRecordStatusQuerier();
}

bool CudaStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CudaOptionalEventRecordStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CudaStreamType::Compute(Instruction* instruction) const {
  OF_PROFILER_RANGE_PUSH("S:" + instruction->instr_msg().DebugName());
  auto* stream = instruction->mut_stream();
  cudaSetDevice(stream->device_id());
  instruction->instr_msg().instruction_type().Compute(instruction);
  OF_CUDA_CHECK(cudaGetLastError());
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  CudaOptionalEventRecordStatusQuerier::MutCast(data_ptr)->SetLaunched(stream->device_ctx().get());
  OF_PROFILER_RANGE_POP();
}

}  // namespace vm
}  // namespace oneflow

#endif
