#include "oneflow/core/operator/cast_op.h"

namespace oneflow {

namespace {

class CastSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastSbpSignatureRule);
  ~CastSbpSignatureRule() override = default;

  CastSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (A,) -> (A,)"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("in");
    if (in_sbp_infer_hint.sbp_parallel().has_split_parallel() == false
        && in_sbp_infer_hint.parallel_num() != parallel_desc.parallel_num()) {
      return MakeSbpSigMatchParallelNumError(parallel_desc.parallel_num(),
                                             in_sbp_infer_hint.parallel_num());
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["in"] = SbpInferHint4BnInOp("in").sbp_parallel();
    (*bn2sbp)["out"] = SbpInferHint4BnInOp("in").sbp_parallel();
  }
};

}  // namespace

void CastOp::InitFromOpConf() {
  CHECK(op_conf().has_cast_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& CastOp::GetCustomizedConf() const { return op_conf().cast_conf(); }

void CastOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->set_data_type(op_conf().cast_conf().data_type());
}

void CastOp::FixSbpSignature(SbpSignature* sbp_signature) const {
  auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
  if (bn2sbp->at("out").has_partial_sum_parallel()) {
    bn2sbp->at("in").mutable_broadcast_parallel();
    bn2sbp->at("out").mutable_broadcast_parallel();
  }
}

void CastOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new CastSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kCastConf, CastOp);

}  // namespace oneflow
