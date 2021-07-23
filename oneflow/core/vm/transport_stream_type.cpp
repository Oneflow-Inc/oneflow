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
#include "oneflow/core/vm/transport_stream_type.h"

namespace oneflow {
namespace vm {

void TransportStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                        Stream* stream) const {
  device_ctx->reset();
}

void TransportStreamType::InitInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(RefCntType) < kInstructionStatusBufferBytes, "");
  new (status_buffer->mut_buffer()->mut_data()) RefCntType(-1);
}

void TransportStreamType::DeleteInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool TransportStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  const char* data = status_buffer.buffer().data();
  return *reinterpret_cast<const RefCntType*>(data) == 0;
}

void TransportStreamType::Compute(Instruction* instruction) const {
  const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
  CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
  instr_type_id.instruction_type().Compute(instruction);
}

template<typename DerivedT>
ObjectMsgPtr<StreamDesc> TransportStreamType::MakeTransportStreamDesc(
    const Resource& resource, int64_t this_machine_id) const {
  std::size_t device_num = 0;
  if (resource.has_cpu_device_num()) {
    device_num = std::max<std::size_t>(device_num, resource.cpu_device_num());
  }
  if (resource.has_gpu_device_num()) {
    device_num = std::max<std::size_t>(device_num, resource.gpu_device_num());
  }
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<DerivedT>());
  // TODO(lixinqi): remove this ugly field
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  // TODO(lixinqi): refactor to a num_threads_per_machine field
  ret->set_num_streams_per_thread(1);
  return ret;
}

ObjectMsgPtr<StreamDesc> TransportSenderStreamType::MakeStreamDesc(const Resource& resource,
                                                                   int64_t this_machine_id) const {
  return MakeTransportStreamDesc<TransportSenderStreamType>(resource, this_machine_id);
}

ObjectMsgPtr<StreamDesc> TransportReceiverStreamType::MakeStreamDesc(
    const Resource& resource, int64_t this_machine_id) const {
  return MakeTransportStreamDesc<TransportReceiverStreamType>(resource, this_machine_id);
}

}  // namespace vm
}  // namespace oneflow
