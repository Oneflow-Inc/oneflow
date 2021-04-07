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
#ifndef ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
#define ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_

#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {
namespace eager {

class LocalCallOpKernelInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override;
  void Compute(vm::Instruction* instruction) const override;

 protected:
  LocalCallOpKernelInstructionType() = default;
  virtual ~LocalCallOpKernelInstructionType() = default;

 private:
  Maybe<void> MaybeInfer(vm::Instruction* instruction) const;
  Maybe<void> MaybeCompute(vm::Instruction* instruction) const;
  virtual const char* device_tag() const = 0;
};

class CallOpKernelInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override;
  void Compute(vm::Instruction* instruction) const override;

 protected:
  CallOpKernelInstructionType() = default;
  virtual ~CallOpKernelInstructionType() = default;

 private:
  Maybe<void> MaybeInfer(vm::Instruction* instruction, const CallOpKernelInstrOperand& args) const;
  Maybe<void> MaybeCompute(vm::Instruction* instruction,
                           const CallOpKernelInstrOperand& args) const;
  virtual const char* device_tag() const = 0;
};

class UserStatelessCallOpKernelInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override;
  void Compute(vm::Instruction* instruction) const override;

 protected:
  UserStatelessCallOpKernelInstructionType() = default;
  virtual ~UserStatelessCallOpKernelInstructionType() = default;

 private:
  Maybe<void> Infer(vm::Instruction* instruction,
                    const StatelessCallOpKernelInstrOperand& args) const;
  Maybe<void> Compute(vm::Instruction* instruction,
                      const StatelessCallOpKernelInstrOperand& args) const;
  virtual const char* device_tag() const = 0;
};

class SystemStatelessCallOpKernelInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override;
  void Compute(vm::Instruction* instruction) const override;

  virtual std::shared_ptr<MemoryCase> GetOutBlobMemCase(const DeviceType device_type,
                                                        const int64_t device_id) const;

 protected:
  SystemStatelessCallOpKernelInstructionType() = default;
  virtual ~SystemStatelessCallOpKernelInstructionType() = default;

 private:
  Maybe<void> Infer(vm::Instruction* instruction,
                    const StatelessCallOpKernelInstrOperand& args) const;
  Maybe<void> Compute(vm::Instruction* instruction,
                      const StatelessCallOpKernelInstrOperand& args) const;
  virtual const char* device_tag() const = 0;
};

class FetchBlobHeaderInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override;
  void Compute(vm::Instruction* instruction) const override {
    // do nothing
  }

 protected:
  FetchBlobHeaderInstructionType() = default;
  virtual ~FetchBlobHeaderInstructionType() = default;

 private:
  virtual const char* device_tag() const = 0;
};

class FetchBlobBodyInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override {
    // do nothing
  }
  void Compute(vm::Instruction* instruction) const override;

 protected:
  FetchBlobBodyInstructionType() = default;
  virtual ~FetchBlobBodyInstructionType() = default;

 private:
  virtual const char* device_tag() const = 0;
};

class FeedBlobInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override {
    // do nothing
  }
  void Compute(vm::Instruction* instruction) const override;

 protected:
  FeedBlobInstructionType() = default;
  virtual ~FeedBlobInstructionType() = default;

 private:
  virtual const char* device_tag() const = 0;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
