#ifndef ONEFLOW_CORE_OPERATOR_LOSS_PRINT_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOSS_PRINT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LossPrintOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossPrintOp);
  LossPrintOp() = default;
  ~LossPrintOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    if (input_bn == "loss_acc") {
      return op_conf().loss_print_conf().loss_lbn();
    } else if (input_bn == "reduction_acc") {
      return op_conf().loss_print_conf().reduction_lbn();
    } else {
      UNIMPLEMENTED();
      return "";
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOSS_PRINT_OP_H_
