#ifndef ONEFLOW_CORE_OPERATOR_VARIABLE_OP_H_
#define ONEFLOW_CORE_OPERATOR_VARIABLE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class VariableOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VariableOp);
  VariableOp()
      : Operator(),
        is_fw_inplace_(std::make_unique<bool>(false)),
        is_bw_inplace_(std::make_unique<bool>(false)) {}
  ~VariableOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  int32_t OutputBlobModelSplitAxis(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const std::string& obn) const override;

  void set_is_fw_inplace(bool val) const { *is_fw_inplace_ = val; }
  void set_is_bw_inplace(bool val) const { *is_bw_inplace_ = val; }

 private:
  void InferSbpSignature(SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
                         const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
                         std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
                         const ParallelDesc& parallel_desc) const override;
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;
  void InferIsModelBlob4OutputBlobs(
      std::function<bool*(const std::string&)> IsModelBlob4BnInOp) const;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;

  std::unique_ptr<bool> is_fw_inplace_;
  std::unique_ptr<bool> is_bw_inplace_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_VARIABLE_OP_H_
