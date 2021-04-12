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
#ifndef ONEFLOW_CORE_VM_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_STREAM_TYPE_H_

#include <string>
#include <typeindex>
#include <glog/logging.h>
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/common/callback.msg.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class ObjectMsgAllocator;

namespace vm {

class Stream;
class InstructionStatusBuffer;
class Instruction;
class VirtualMachine;
class InstructionMsg;
class InstructionType;

class StreamType {
 public:
  virtual ~StreamType() = default;

  void Run(Instruction* instruction) const;
  void Run(VirtualMachine* vm, InstructionMsg* instr_msg) const;
  void Run(VirtualMachine* vm, Instruction* instruction) const;

  virtual const char* device_tag() const = 0;

  virtual void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const = 0;

  virtual void InitInstructionStatus(const Stream& stream,
                                     InstructionStatusBuffer* status_buffer) const = 0;
  virtual void DeleteInstructionStatus(const Stream& stream,
                                       InstructionStatusBuffer* status_buffer) const = 0;
  virtual bool QueryInstructionStatusDone(const Stream& stream,
                                          const InstructionStatusBuffer& status_buffer) const = 0;
  virtual void Compute(Instruction* instruction) const = 0;
  virtual void Infer(Instruction* instruction) const { LOG(FATAL) << "UNIMPLEMENTED"; }

  virtual ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                                  int64_t this_machine_id) const = 0;

  virtual bool SharingVirtualMachineThread() const = 0;
  virtual bool IsControlStreamType() const { return false; }
  virtual void Infer(VirtualMachine* vm, Instruction* instruction) const { Infer(instruction); }
  virtual void Compute(VirtualMachine* vm, Instruction* instruction) const { Compute(instruction); }
  virtual void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  virtual void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }

 protected:
  StreamType() = default;
};

HashMap<std::type_index, const StreamType*>* StreamType4TypeIndex();

template<typename T>
const StreamType* LookupStreamType4TypeIndex() {
  return StreamType4TypeIndex()->at(typeid(T));
}

template<typename T>
void TryRegisterStreamType4TypeIndex() {
  auto* map = StreamType4TypeIndex();
  std::type_index type_index(typeid(T));
  if (map->find(type_index) == map->end()) { map->emplace(type_index, new T()); }
}

const StreamTypeId& LookupInferStreamTypeId(const StreamTypeId& compute_stream_type_id);
void TryRegisterInferStreamTypeId(const StreamType* infer_stream_type,
                                  const StreamType* compute_stream_type);
template<typename InferStreamType, typename ComputeStreamType>
void TryRegisterInferStreamTypeId() {
  TryRegisterInferStreamTypeId(LookupStreamType4TypeIndex<InferStreamType>(),
                               LookupStreamType4TypeIndex<ComputeStreamType>());
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_TYPE_H_
