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

#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"

namespace oneflow {
namespace vm {

void CudaCopyH2DStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                          Stream* stream) const {
  device_ctx->reset(new CudaStreamHandleDeviceCtx(stream->device_id()));
}

void CudaCopyH2DStreamType::InitInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  auto* event_provider = dynamic_cast<QueryCudaEventProvider*>(stream.device_ctx().get());
  auto* data_ptr = status_buffer->mut_buffer()->mut_data();
  const auto& cuda_event = CHECK_NOTNULL(event_provider)->GetCudaEvent();
  CudaOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, cuda_event);
}

void CudaCopyH2DStreamType::DeleteInstructionStatus(const Stream& stream,
                                                    InstructionStatusBuffer* status_buffer) const {
  auto* ptr =
      CudaOptionalEventRecordStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data());
  ptr->~CudaOptionalEventRecordStatusQuerier();
}

bool CudaCopyH2DStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CudaOptionalEventRecordStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CudaCopyH2DStreamType::Compute(Instruction* instruction) const {
  auto* stream = instruction->mut_stream();
  cudaSetDevice(stream->device_id());
  instruction->instr_msg().instruction_type().Compute(instruction);
  OF_CUDA_CHECK(cudaGetLastError());
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  CudaOptionalEventRecordStatusQuerier::MutCast(data_ptr)->SetLaunched(stream->device_ctx().get());
}

}  // namespace vm
}  // namespace oneflow

#endif
