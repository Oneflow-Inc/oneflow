#ifndef ONEFLOW_CORE_OPERATOR_LAYER_NORM_OP_H_
#define ONEFLOW_CORE_OPERATOR_LAYER_NORM_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LayerNormOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LayerNormOp);
  LayerNormOp() = default;
  ~LayerNormOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().layer_norm_conf(); }
  bool NeedInBlobWhenBackward() const override { return true; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override;
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;

  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LAYER_NORM_OP_H_
