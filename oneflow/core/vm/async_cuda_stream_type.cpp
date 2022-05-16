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

#include "oneflow/core/vm/async_cuda_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/cuda_stream_handle_device_context.h"
#include "oneflow/core/vm/cuda_optional_event_record_status_querier.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

void AsyncCudaStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                        Stream* stream) const {
  device_ctx->reset(new CudaStreamHandleDeviceCtx(stream->device_id()));
}

void AsyncCudaStreamType::InitInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  auto* event_provider = dynamic_cast<QueryCudaEventProvider*>(stream.device_ctx().get());
  auto* data_ptr = status_buffer->mut_buffer()->mut_data();
  const auto& cuda_event = CHECK_NOTNULL(event_provider)->GetCudaEvent();
  CudaOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, cuda_event);
}

void AsyncCudaStreamType::DeleteInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  auto* ptr =
      CudaOptionalEventRecordStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data());
  ptr->~CudaOptionalEventRecordStatusQuerier();
}

bool AsyncCudaStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CudaOptionalEventRecordStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void AsyncCudaStreamType::Compute(Instruction* instruction) const {
  OF_PROFILER_RANGE_GUARD("S:" + instruction->instr_msg().DebugName());
  auto* stream = instruction->mut_stream();
  cudaSetDevice(stream->device_id());
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    instr_type_id.instruction_type().Compute(instruction);
    OF_CUDA_CHECK(cudaGetLastError());
  }
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  CudaOptionalEventRecordStatusQuerier::MutCast(data_ptr)->SetLaunched(stream->device_ctx().get());
}

intrusive::shared_ptr<StreamDesc> AsyncCudaStreamType::MakeStreamDesc(
    const Resource& resource, int64_t this_machine_id) const {
  if (!resource.has_gpu_device_num()) { return intrusive::shared_ptr<StreamDesc>(); }
  std::size_t device_num = resource.gpu_device_num();
  auto ret = intrusive::make_shared<StreamDesc>();
  ret->set_stream_type(StaticGlobalStreamType<AsyncCudaStreamType>());
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(device_num);
  return ret;
}

}  // namespace vm
}  // namespace oneflow

#endif
