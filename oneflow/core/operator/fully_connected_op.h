#ifndef ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_
#define ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class FullyConnectedOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FullyConnectedOp);
  FullyConnectedOp() = default;
  ~FullyConnectedOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
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
  void GetSbpSignatureRules(std::vector<std::unique_ptr<const SbpSignatureRule>>*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_
