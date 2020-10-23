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
#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class NopStreamType final : public StreamType {
 public:
  NopStreamType() = default;
  ~NopStreamType() = default;

  const char* device_tag() const override { return "cpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Compute(Instruction* instruction) const override;
  ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                          int64_t this_machine_id) const override;
};

class NopInstructionType final : public InstructionType {
 public:
  NopInstructionType() = default;
  ~NopInstructionType() override = default;

  using stream_type = NopStreamType;

  void Infer(Instruction* instruction) const override { /* do nothing */
  }
  void Compute(Instruction* instruction) const override { UNIMPLEMENTED(); }
};
COMMAND(RegisterInstructionType<NopInstructionType>("Nop"));

void NopStreamType::InitInstructionStatus(const Stream& stream,
                                          InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void NopStreamType::DeleteInstructionStatus(const Stream& stream,
                                            InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool NopStreamType::QueryInstructionStatusDone(const Stream& stream,
                                               const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void NopStreamType::Compute(Instruction* instruction) const {
  auto* status_buffer = instruction->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

ObjectMsgPtr<StreamDesc> NopStreamType::MakeStreamDesc(const Resource& resource,
                                                       int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<NopStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
