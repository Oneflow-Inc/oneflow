#ifndef ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/instruction_operand.msg.h"

namespace oneflow {

namespace one {

class TensorTuple;
class StatefullOpKernel;

}

namespace user_op {

class InferContext;

}

namespace eager {

class LocalCallOpKernelPhyInstrOperand final : public vm::PhyInstrOperand {
 public:
  LocalCallOpKernelPhyInstrOperand(const LocalCallOpKernelPhyInstrOperand&) = delete;
  LocalCallOpKernelPhyInstrOperand(LocalCallOpKernelPhyInstrOperand&&) = delete;
  ~LocalCallOpKernelPhyInstrOperand() override = default;

  LocalCallOpKernelPhyInstrOperand(
    const std::shared_ptr<StatefullOpKernel>& opkernel,
    const std::shared_ptr<TensorTuple>& inputs,
    const std::shared_ptr<TensorTuple>& outputs):
      opkernel_(opkernel), inputs_(inputs), outputs_(outputs) {}

  const StatefullOpKernel& opkernel() const { return *opkernel_; }
  const TensorTuple& inputs() const { return *inputs_; }
  const TensorTuple& outputs() const { return *outputs_; }

  StatefullOpKernel* mut_opkernel() { return opkernel_.get(); }
  const std::shared_ptr<TensorTuple>& mut_inputs() { return inputs_; }
  const std::shared_ptr<TensorTuple>& mut_outputs() { return inputs_; }

 private:
  std::shared_ptr<StatefullOpKernel> opkernel_;
  std::shared_ptr<TensorTuple> inputs_;
  std::shared_ptr<TensorTuple> outputs_;
}; 

}
}

#endif  // ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
