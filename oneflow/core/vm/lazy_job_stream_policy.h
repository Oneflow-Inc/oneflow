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

#ifndef ONEFLOW_CORE_VM_LAZY_JOB_STREAM_POLICY_H_
#define ONEFLOW_CORE_VM_LAZY_JOB_STREAM_POLICY_H_

#include "oneflow/core/vm/stream_policy.h"
#include "oneflow/core/vm/instruction.h"

namespace oneflow {

class NNGraphIf;

namespace vm {

class LazyJobStreamPolicy final : public StreamPolicy {
 public:
  LazyJobStreamPolicy() = default;
  virtual ~LazyJobStreamPolicy() = default;

  vm::Allocator* mut_allocator() override { return (vm::Allocator*)nullptr; }

  DeviceType device_type() const override {
    UNIMPLEMENTED();
    return DeviceType::kInvalidDevice;
  }

  ep::Stream* stream() override {
    UNIMPLEMENTED();
    return nullptr;
  }

  void WaitUntilQueueEmptyIfFrontNNGraphNotEquals(const std::shared_ptr<NNGraphIf>& nn_graph);

  void EnqueueNNGraph(const std::shared_ptr<NNGraphIf>& nn_graph);

  void DequeueNNGraph();

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusLaunched(const Stream& stream,
                                      const InstructionStatusBuffer& status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Run(Instruction* instruction) const override;
  bool SupportingTransportInstructions() const override { return false; }

 private:
  std::queue<std::weak_ptr<NNGraphIf>> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LAZY_JOB_STREAM_POLICY_H_
