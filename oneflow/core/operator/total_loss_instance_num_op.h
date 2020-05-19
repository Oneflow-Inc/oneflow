#ifndef ONEFLOW_CORE_OPERATOR_TOTAL_LOSS_INSTANCE_NUM_OP_H_
#define ONEFLOW_CORE_OPERATOR_TOTAL_LOSS_INSTANCE_NUM_OP_H_

#include "oneflow/core/operator/cwise_op.h"

namespace oneflow {

class TotalLossInstanceNumOp final : public CWiseOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TotalLossInstanceNumOp);
  TotalLossInstanceNumOp() = default;
  ~TotalLossInstanceNumOp() = default;

  void VirtualInitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_TOTAL_LOSS_INSTANCE_NUM_OP_H_
