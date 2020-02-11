#ifndef ONEFLOW_CORE_OPERATOR_BROADCAST_DIV_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_BROADCAST_DIV_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BroadcastDivGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastDivGradOp);
  BroadcastDivGradOp() = default;
  ~BroadcastDivGradOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BROADCAST_DIV_GRAD_OP_H_
