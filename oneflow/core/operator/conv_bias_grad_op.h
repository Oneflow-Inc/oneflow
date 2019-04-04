#ifndef ONEFLOW_CORE_OPERATOR_CONV_BIAS_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_BIAS_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ConvBiasGradOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvBiasGradOp);
  ConvBiasGradOp() = default;
  ~ConvBiasGradOp() override = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  int32_t OutputBlobModelSplitAxis(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const std::string& obn) const override;

 private:
  const PbMessage& GetCustomizedConf() const override;
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void GetSbpSignatures(std::vector<std::unique_ptr<const SbpSignature>>*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_BIAS_GRAD_OP_H_
