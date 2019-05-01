#ifndef ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NormalModelUpdtOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalModelUpdtOp);
  virtual ~NormalModelUpdtOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  virtual const PbMessage& GetCustomizedConf() const override;

 protected:
  NormalModelUpdtOp() = default;
  virtual void MdUpdtVirtualInitFromOpConf() {}
  virtual void MdUpdtVirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*) const {}

  virtual const HashSet<std::string> AlwaysBroadcastParallelBns() const = 0;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;

  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

std::unique_ptr<const SbpSignatureRule> MakeModelUpdtOpBroadcastSignatureRule(const Operator*);
std::unique_ptr<const SbpSignatureRule> MakeModelUpdtOpSplitSignatureRule(
    const Operator*, const HashSet<std::string>& always_broadcast_parallel_bns);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_
