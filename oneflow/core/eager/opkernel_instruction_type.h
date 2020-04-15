#ifndef ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
#define ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_

#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"

namespace oneflow {
namespace eager {

class CallOpKernelInstructionType : public vm::InstructionType {
 public:
  void Infer(vm::InstrCtx* instr_ctx) const override;
  void Compute(vm::InstrCtx* instr_ctx) const override;

 protected:
  CallOpKernelInstructionType() = default;
  virtual ~CallOpKernelInstructionType() = default;

 private:
  virtual const char* device_tag() const = 0;

  template<typename T>
  void Infer(vm::InstrCtx* instr_ctx, const T& args) const;
  template<typename T>
  void Compute(vm::InstrCtx* instr_ctx, const T& args) const;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
