#ifndef ONEFLOW_CORE_OPERATOR_BIAS_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_BIAS_ADD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BiasAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BiasAddOp);
  BiasAddOp() = default;
  ~BiasAddOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  int32_t OutputBlobModelSplitAxis(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const std::string& obn) const override {
    return 1;
  }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override;
  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BIAS_ADD_OP_H_
