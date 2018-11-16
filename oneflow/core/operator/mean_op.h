#ifndef ONEFLOW_CORE_OPERATOR_MEAN_OP_H_
#define ONEFLOW_CORE_OPERATOR_MEAN_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class MeanOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MeanOp);
  MeanOp() = default;
  ~MeanOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().mean_conf(); }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*) const override;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MEAN_OP_H_
