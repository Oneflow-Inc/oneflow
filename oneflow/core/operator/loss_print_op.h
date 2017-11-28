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
  const PbMessage& GetSpecialConf() const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kPackedBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOSS_PRINT_OP_H_
