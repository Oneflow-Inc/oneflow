#ifndef ONEFLOW_CORE_OPERATOR_ELEMENTWISE_OP_H_
#define ONEFLOW_CORE_OPERATOR_ELEMENTWISE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ElementwiseOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseOp);
  ElementwiseOp() = default;
  ~ElementwiseOp() = default;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  void InitFromOpConf() override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ELEMENTWISE_OP_H_
