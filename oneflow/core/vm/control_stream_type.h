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
#ifndef ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_

#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"

namespace oneflow {
namespace vm {

class VirtualMachine;
class InstructionMsg;

class ControlStreamType final : public StreamType {
 public:
  ControlStreamType() = default;
  ~ControlStreamType() = default;

  const char* device_tag() const override { return "cpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                          int64_t this_machine_id) const override;
  void Compute(Instruction* instruction) const override;

  bool SharingVirtualMachineThread() const override { return true; }
  bool IsControlStreamType() const override { return true; }
  void Infer(VirtualMachine* vm, Instruction* instruction) const override;
  void Compute(VirtualMachine* vm, Instruction* instruction) const override;
  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override;
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_VM_STREAM_TYPE_H_
