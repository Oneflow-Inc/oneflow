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
  device_ctx->reset(
      new CudaStreamHandleDeviceCtx(stream->mut_callback_list(), stream->device_id()));
}

void CudaCopyH2DStreamType::InitInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  CudaInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data(), stream.device_id());
}

void CudaCopyH2DStreamType::DeleteInstructionStatus(const Stream& stream,
                                                    InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool CudaCopyH2DStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CudaInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CudaCopyH2DStreamType::Compute(Instruction* instruction) const {
  auto* stream = instruction->mut_stream();
  cudaSetDevice(stream->device_id());
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    instr_type_id.instruction_type().Compute(instruction);
    OF_CUDA_CHECK(cudaGetLastError());
  }
  stream->mut_callback_list()->MoveTo(instruction->mut_callback_list());
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  CudaInstrStatusQuerier::MutCast(data_ptr)->SetLaunched(stream->device_ctx().get());
}

ObjectMsgPtr<StreamDesc> CudaCopyH2DStreamType::MakeStreamDesc(const Resource& resource,
                                                               int64_t this_machine_id) const {
  if (!resource.has_gpu_device_num()) { return ObjectMsgPtr<StreamDesc>(); }
  std::size_t device_num = resource.gpu_device_num();
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<CudaCopyH2DStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(1);
  return ret;
}

}  // namespace vm
}  // namespace oneflow

#endif
