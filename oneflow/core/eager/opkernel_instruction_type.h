#ifndef ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
#define ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_

#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {
namespace eager {

class CallOpKernelInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override;
  void Compute(vm::Instruction* instruction) const override;

 protected:
  CallOpKernelInstructionType() = default;
  virtual ~CallOpKernelInstructionType() = default;

 private:
  Maybe<void> MaybeInfer(vm::Instruction* instruction) const;
  Maybe<void> MaybeCompute(vm::Instruction* instruction) const;
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
  Maybe<void> MaybeInfer(vm::Instruction* instruction) const;
  Maybe<void> MaybeCompute(vm::Instruction* instruction) const;
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
  Maybe<void> MaybeInfer(vm::Instruction* instruction) const;
  Maybe<void> MaybeCompute(vm::Instruction* instruction) const;
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
