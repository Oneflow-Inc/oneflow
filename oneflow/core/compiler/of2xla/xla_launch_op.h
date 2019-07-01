#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_OP_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_OP_H_

#include <unordered_map>
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class XlaLaunchOp : public Operator {
 public:
  void InitFromOpConf() override;
  
  const PbMessage& GetCustomizedConf() const override;

  bool NeedInBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;

  void InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const override;

  std::unordered_map<std::string, std::string> subgraph_inputs_;
  std::unordered_map<std::string, std::string> subgraph_outputs_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_OP_H_
