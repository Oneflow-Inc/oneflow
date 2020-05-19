#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

class DistributeAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DistributeAddOp);
  DistributeAddOp() = default;
  ~DistributeAddOp() = default;

  void InitFromOpConf() override;

  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override;
  LogicalNode* NewProperLogicalNode() const override { return new DistributeConcatLogicalNode; }

 private:
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
};

void DistributeAddOp::InitFromOpConf() {
  CHECK(op_conf().has_distribute_add_conf());

  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& DistributeAddOp::GetCustomizedConf() const {
  return op_conf().distribute_add_conf();
}

Maybe<void> DistributeAddOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* first_blob_desc = nullptr;
  FOR_RANGE(int, i, 0, input_bns().size()) {
    first_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    if (first_blob_desc != nullptr) { break; }
  }
  CHECK_NOTNULL(first_blob_desc);
  *GetBlobDesc4BnInOp("out") = *first_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> DistributeAddOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  FOR_RANGE(int32_t, i, 0, input_bns().size()) {
    CHECK_OR_RETURN(*BatchAxis4BnInOp(input_bns().Get(i)) == *BatchAxis4BnInOp(input_bns().Get(0)));
  }
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp(input_bns().Get(0));
  return Maybe<void>::Ok();
}

Maybe<void> DistributeAddOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), input_bns().size());
  const auto& first_in_hint = *JUST(SbpInferHint4Ibn(input_bns().Get(0)));
  FOR_RANGE(int, i, 0, input_bns().size()) {
    const auto& in_sbp_infer_hint = *JUST(SbpInferHint4Ibn(input_bns().Get(i)));
    CHECK_EQ_OR_RETURN(1, in_sbp_infer_hint.parallel_desc().parallel_num());
    CHECK_EQ_OR_RETURN(first_in_hint.logical_blob_desc().shape(),
                       in_sbp_infer_hint.logical_blob_desc().shape());
  }
  auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
  for (const auto& ibn : input_bns()) { (*bn2sbp)[ibn].mutable_partial_sum_parallel(); }
  (*bn2sbp)["out"].mutable_partial_sum_parallel();
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDistributeAddConf, DistributeAddOp);
REGISTER_DISABLE_INPUT_BOXING_GROUP(OperatorConf::kDistributeAddConf);

}  // namespace oneflow
