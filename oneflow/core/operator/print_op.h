#ifndef ONEFLOW_CORE_OPERATOR_PRINT_OP_H_
#define ONEFLOW_CORE_OPERATOR_PRINT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PrintOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrintOp);
  PrintOp() = default;
  ~PrintOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  bool IsPrintOp() const override { return true; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {}

 private:
  std::string ibn2lbn(const std::string& ibn) const override {
    return ibn2lbn_.at(ibn);
  }

  HashMap<std::string, std::string> ibn2lbn_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PRINT_OP_H_
