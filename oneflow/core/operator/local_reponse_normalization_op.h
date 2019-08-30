#ifndef ONEFLOW_CORE_OPERATOR_LOCAL_REPONSE_NORMALIZATION_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOCAL_REPONSE_NORMALIZATION_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LocalResponseNormalizationOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalResponseNormalizationOp);
  LocalResponseNormalizationOp() = default;
  ~LocalResponseNormalizationOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOCAL_REPONSE_NORMALIZATION_OP_H_
