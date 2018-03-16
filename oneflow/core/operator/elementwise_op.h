#ifndef ONEFLOW_CORE_OPERATOR_ELEMENTWISE_OP_H_
#define ONEFLOW_CORE_OPERATOR_ELEMENTWISE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ElementwiseOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseOp);
  ElementwiseOp() = default;
  virtual ~ElementwiseOp() = default;

  void InitFromOpConf() override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 protected:
 protected:
  virtual void VirtualInitFromOpConf() { UNIMPLEMENTED(); }

  virtual void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ELEMENTWISE_OP_H_
