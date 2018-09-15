#ifndef ONEFLOW_CORE_OPERATOR_AFFINE_CHANNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_AFFINE_CHANNEL_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AffineChannelOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AffineChannelOp);
  AffineChannelOp() = default;
  ~AffineChannelOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_AFFINE_CHANNEL_OP_H_
