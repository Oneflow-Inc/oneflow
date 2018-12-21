#ifndef ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_
#define ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReshapeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReshapeOp);
  ReshapeOp() = default;
  ~ReshapeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsElemWiseOp() const override { return true; }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_
