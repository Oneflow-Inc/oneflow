#ifndef ONEFLOW_CORE_OPERATOR_V_STACK_OP_H_
#define ONEFLOW_CORE_OPERATOR_V_STACK_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class VStackOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VStackOp);
  VStackOp() = default;
  ~VStackOp() = default;

  void InitFromOpConf() override;

  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_VSTACK_OP_H_
