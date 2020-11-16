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
#ifndef ONEFLOW_CORE_EAGER_SEND_BLOB_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_EAGER_SEND_BLOB_INSTRUCTION_TYPE_H_

#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/transport_stream_type.h"

namespace oneflow {
namespace eager {

class SendBlobInstructionType : public vm::InstructionType {
 public:
  SendBlobInstructionType() = default;
  virtual ~SendBlobInstructionType() override = default;

  using stream_type = vm::TransportSenderStreamType;
  using RefCntType = vm::TransportSenderStreamType::RefCntType;

  void Infer(vm::Instruction* instruction) const override {
    // do nothing
  }
  void Compute(vm::Instruction* instruction) const override;

 protected:
  virtual Maybe<void> Send(int64_t dst_machine_id, uint64_t token, const char* mem_ptr,
                           std::size_t size, const std::function<void()>& Callback) const;

 private:
  Maybe<void> Send(vm::Instruction* instruction) const;
};

class ReceiveBlobInstructionType : public vm::InstructionType {
 public:
  ReceiveBlobInstructionType() = default;
  virtual ~ReceiveBlobInstructionType() override = default;

  using stream_type = vm::TransportSenderStreamType;
  using RefCntType = vm::TransportSenderStreamType::RefCntType;

  void Infer(vm::Instruction* instruction) const override {
    // do nothing
  }
  void Compute(vm::Instruction* instruction) const override;

 protected:
  virtual Maybe<void> Receive(int64_t src_machine_id, uint64_t token, char* mem_ptr,
                              std::size_t size, const std::function<void()>& Callback) const;

 private:
  Maybe<void> Receive(vm::Instruction* instruction) const;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_SEND_BLOB_INSTRUCTION_TYPE_H_
