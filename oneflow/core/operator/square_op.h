#ifndef ONEFLOW_CORE_OPERATOR_SQUARE_H_
#define ONEFLOW_CORE_OPERATOR_SQUARE_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SquareOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SquareOp);
  SquareOp() = default;
  ~SquareOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsElemWiseOp() const override { return true; }
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SQUARE_H_
