#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ReduceConcatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceConcatOp);
  ReduceConcatOp() = default;
  ~ReduceConcatOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  LogicalNode* NewProperLogicalNode() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void InferSbpSignature(SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
                         const SbpSignature& sbp_sig_hint,
                         const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
                         std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
                         const ParallelDesc& parallel_desc) const override;
  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*,
                            const OpContext* op_ctx) const override;
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_
