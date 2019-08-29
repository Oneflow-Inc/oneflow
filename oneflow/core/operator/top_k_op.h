#ifndef ONEFLOW_CORE_OPERATOR_TOP_K_OP_H_
#define ONEFLOW_CORE_OPERATOR_TOP_K_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class TopKOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TopKOp);
  TopKOp() = default;
  ~TopKOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().top_k_conf(); }

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_TOP_K_OP_H_
