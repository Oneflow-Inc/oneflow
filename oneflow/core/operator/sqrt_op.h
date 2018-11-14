#ifndef ONEFLOW_CORE_OPERATOR_SQRT_H_
#define ONEFLOW_CORE_OPERATOR_SQRT_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SqrtOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SqrtOp);
  SqrtOp() = default;
  ~SqrtOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsElemWiseOp() const override { return true; }
  bool NeedInBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SQRT_H_
