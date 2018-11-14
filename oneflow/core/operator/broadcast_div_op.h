#ifndef ONEFLOW_CORE_OPERATOR_BROADCAST_DIV_OP_H_
#define ONEFLOW_CORE_OPERATOR_BROADCAST_DIV_OP_H_

#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastDivOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastDivOp);
  BroadcastDivOp() = default;
  ~BroadcastDivOp() = default;

  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }

 private:
  void DerivedInitFromOpConf() override;
  void DerivedInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BROADCAST_DIV_OP_H_
