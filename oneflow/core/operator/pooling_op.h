#ifndef ONEFLOW_CORE_OPERATOR_POOLING_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PoolingOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingOp);
  PoolingOp() = default;
  ~PoolingOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraActivationDiffMem() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  const PbMessage& GetSpecialConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_OP_H_
