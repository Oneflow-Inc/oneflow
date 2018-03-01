#ifndef ONEFLOW_CORE_OPERATOR_SOFTMAX_OP_H_
#define ONEFLOW_CORE_OPERATOR_SOFTMAX_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SoftmaxOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxOp);
  SoftmaxOp() = default;
  ~SoftmaxOp() = default;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      DevictType device_type,
    std::function<void(OpContext*)> EnrollOpContext
      ) const override;

 private:
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*, const OpContext* op_ctx, KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SOFTMAX_OP_H_
