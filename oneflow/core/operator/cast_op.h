#ifndef ONEFLOW_CORE_OPERATOR_CAST_OP_H_
#define ONEFLOW_CORE_OPERATOR_CAST_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CastOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastOp);
  CastOp() = default;
  ~CastOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void FixSbpSignature(SbpSignature* sbp_signature) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return true; }
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CAST_OP_H_
