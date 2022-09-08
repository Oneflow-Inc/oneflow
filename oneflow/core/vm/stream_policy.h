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
#ifndef ONEFLOW_CORE_VM_STREAM_POLICY_H_
#define ONEFLOW_CORE_VM_STREAM_POLICY_H_

#include <string>
#include <typeindex>
#include <glog/logging.h>
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class EpEventProvider;

namespace ep {

class Device;
class Stream;

}  // namespace ep

namespace vm {

class Allocator;
class Stream;
class InstructionStatusBuffer;
class Instruction;

class StreamPolicy {
 public:
  virtual ~StreamPolicy() = default;

  virtual ep::Stream* stream() = 0;
  virtual vm::Allocator* mut_allocator() = 0;
  virtual DeviceType device_type() const = 0;

  virtual void InitInstructionStatus(const Stream& stream,
                                     InstructionStatusBuffer* status_buffer) const = 0;
  virtual void DeleteInstructionStatus(const Stream& stream,
                                       InstructionStatusBuffer* status_buffer) const = 0;
  virtual bool QueryInstructionStatusLaunched(
      const Stream& stream, const InstructionStatusBuffer& status_buffer) const = 0;
  virtual bool QueryInstructionStatusDone(const Stream& stream,
                                          const InstructionStatusBuffer& status_buffer) const = 0;
  virtual bool OnSchedulerThread(StreamType stream_type) const;
  virtual bool SupportingTransportInstructions() const = 0;

  void RunIf(Instruction* instruction) const;

 protected:
  StreamPolicy() = default;

 private:
  virtual void Run(Instruction* instruction) const = 0;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_POLICY_H_
