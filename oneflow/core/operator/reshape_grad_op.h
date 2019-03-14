#ifndef ONEFLOW_CORE_OPERATOR_RESHAPE_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_RESHAPE_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/reshape_op.h"

namespace oneflow {

class ReshapeGradOp final : public ReshapeOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReshapeGradOp);
  ReshapeGradOp() = default;
  ~ReshapeGradOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsForwardInplace() const override {
    return false;
  }  // It could be inplace but won't pass the check for sole input

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RESHAPE_GRAD_OP_H_
