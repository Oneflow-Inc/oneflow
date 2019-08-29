#ifndef ONEFLOW_CORE_OPERATOR_CONV_FILTER_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_FILTER_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ConvFilterGradOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradOp);
  ConvFilterGradOp() = default;
  ~ConvFilterGradOp() override = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, int64_t record_piece_size,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  const PbMessage& GetCustomizedConf() const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_FILTER_GRAD_OP_H_
