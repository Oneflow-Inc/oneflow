#ifndef ONEFLOW_CORE_OPERATOR_LAYER_NORM_PARAM_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_LAYER_NORM_PARAM_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LayerNormParamGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LayerNormParamGradOp);
  LayerNormParamGradOp() = default;
  ~LayerNormParamGradOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().layer_norm_param_grad_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LAYER_NORM_PARAM_GRAD_OP_H_
