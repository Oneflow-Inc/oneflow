#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

class DistributeCloneOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeCloneOp);
  DistributeCloneOp() = default;
  ~DistributeCloneOp() = default;

  void InitFromOpConf() override;

  const PbMessage& GetCustomizedConf() const override;

  LogicalNode* NewProperLogicalNode() const override { return new DistributeSplitLogicalNode; }

 private:
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override;
  Maybe<void> InferOutParallelDesc(
      std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn,
      std::function<const BlobDesc*(const std::string&)> LogicalBlobDesc4Ibn, const ParallelDesc&,
      const SbpSignature*) const override;
};

void DistributeCloneOp::InitFromOpConf() {
  CHECK(op_conf().has_distribute_clone_conf());

  EnrollInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& DistributeCloneOp::GetCustomizedConf() const {
  return op_conf().distribute_clone_conf();
}

Maybe<void> DistributeCloneOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& in_blob_desc = *GetBlobDesc4BnInOp("in");
  if (parallel_ctx->parallel_num() > 1) {
    CHECK_EQ(parallel_ctx->parallel_num(), output_bns().size());
    auto* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(parallel_ctx->parallel_id()));
    *out_blob_desc = in_blob_desc;
    return Maybe<void>::Ok();
  }
  FOR_RANGE(int, i, 0, output_bns().size()) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    if (blob_desc != nullptr) { *blob_desc = in_blob_desc; }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeCloneOp::InferOutParallelDesc(
    std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn,
    std::function<const BlobDesc*(const std::string&)> LogicalBlobDesc4Ibn,
    const ParallelDesc& op_parallel_desc, const SbpSignature*) const {
  FOR_RANGE(int, i, 0, output_bns().size()) {
    const auto& obn = output_bns().Get(i);
    if (op_parallel_desc.parallel_num() > 1) {
      CHECK_EQ(op_parallel_desc.parallel_num(), output_bns().size());
      *ParallelDesc4Obn(obn) = ParallelDesc(op_parallel_desc.GetParallelIdOnlyParallelConf(i));
    } else {
      *ParallelDesc4Obn(obn) = op_parallel_desc;
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeCloneOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  FOR_RANGE(int32_t, i, 0, output_bns().size()) {
    *BatchAxis4BnInOp(output_bns().Get(i)) = *BatchAxis4BnInOp("in");
  }
  return Maybe<void>::Ok();
}

Maybe<void> DistributeCloneOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), output_bns().size());
  const SbpInferHint& in_hint = *JUST(SbpInferHint4Ibn("in"));
  CHECK_OR_RETURN(in_hint.parallel_desc() == parallel_desc);
  SbpSignatureBuilder().Broadcast(output_bns()).Build(sbp_signature);
  auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
  (*bn2sbp)["in"] = in_hint.sbp_parallel();
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDistributeCloneConf, DistributeCloneOp);

}  // namespace oneflow
