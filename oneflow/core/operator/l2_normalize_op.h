#ifndef ONEFLOW_CORE_OPERATOR_L2_NORMALIZE_OP_H_
#define ONEFLOW_CORE_OPERATOR_L2_NORMALIZE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class L2NormalizeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeOp);
  L2NormalizeOp() = default;
  ~L2NormalizeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_L2_NORMALIZE_OP_H_
