#ifndef ONEFLOW_CORE_OPERATOR_CONV_DATA_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_DATA_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ConvDataGradOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradOp);
  ConvDataGradOp() = default;
  ~ConvDataGradOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  const PbMessage& GetCustomizedConf() const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_DATA_GRAD_OP_H_
