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
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

namespace vm {

class Stream;
class InstructionStatusBuffer;
class Instruction;
class InstructionType;

class StreamType {
 public:
  virtual ~StreamType() = default;

  void Run(Instruction* instruction) const { Compute(instruction); }

  virtual void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const = 0;

  virtual void InitInstructionStatus(const Stream& stream,
                                     InstructionStatusBuffer* status_buffer) const = 0;
  virtual void DeleteInstructionStatus(const Stream& stream,
                                       InstructionStatusBuffer* status_buffer) const = 0;
  virtual bool QueryInstructionStatusDone(const Stream& stream,
                                          const InstructionStatusBuffer& status_buffer) const = 0;
  virtual void Compute(Instruction* instruction) const = 0;

  virtual bool OnSchedulerThread() const = 0;
  virtual bool SupportingTransportInstructions() const = 0;
  virtual bool IsControlStreamType() const { return false; }

 protected:
  StreamType() = default;
};

template<typename T>
const StreamType* StaticGlobalStreamType() {
  static const StreamType* stream_type = new T();
  return stream_type;
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_TYPE_H_
