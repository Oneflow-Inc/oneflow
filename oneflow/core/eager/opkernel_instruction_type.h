#ifndef ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
#define ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_

#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"

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
  virtual const char* device_tag() const = 0;
};

class StatelessCallOpKernelInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::Instruction* instruction) const override;
  void Compute(vm::Instruction* instruction) const override;

 protected:
  StatelessCallOpKernelInstructionType() = default;
  virtual ~StatelessCallOpKernelInstructionType() = default;

 private:
  virtual const char* device_tag() const = 0;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
