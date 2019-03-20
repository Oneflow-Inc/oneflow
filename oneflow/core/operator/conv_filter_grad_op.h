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
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  int32_t OutputBlobModelSplitAxis(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const std::string& obn) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
  void GetOpParallelSignatures(
      std::vector<std::unique_ptr<const OpParallelSignature>>*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_FILTER_GRAD_OP_H_
