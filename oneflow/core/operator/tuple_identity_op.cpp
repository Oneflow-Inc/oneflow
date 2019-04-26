#include "oneflow/core/operator/tuple_identity_op.h"

namespace oneflow {

namespace {

class TupleIdentitySbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TupleIdentitySbpSignatureRule);
  ~TupleIdentitySbpSignatureRule() override = default;

  TupleIdentitySbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": A -> A"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const auto& bn2conf_sbp = op().op_conf().sbp_signature_hint().bn_in_op2sbp_parallel();
    FOR_RANGE(int32_t, i, 0, op().input_bns().size()) {
      const SbpParallel* sbp_parallel = nullptr;
      const auto& conf_sbp_it = bn2conf_sbp.find(op().output_bns().Get(i));
      if (conf_sbp_it == bn2conf_sbp.end()) {
        sbp_parallel = &SbpInferHint4BnInOp(op().input_bns().Get(i)).sbp_parallel();
      } else {
        sbp_parallel = &conf_sbp_it->second;
      }
      (*bn2sbp)[op().input_bns().Get(i)] = *sbp_parallel;
      (*bn2sbp)[op().output_bns().Get(i)] = *sbp_parallel;
    }
  }
};

}  // namespace

void TupleIdentityOp::InitFromOpConf() {
  CHECK(op_conf().has_tuple_identity_conf());
  int32_t in_size = op_conf().tuple_identity_conf().in_size();
  int32_t out_size = op_conf().tuple_identity_conf().out_size();
  CHECK_GT(in_size, 0);
  CHECK_EQ(in_size, out_size);
  EnrollRepeatedInputBn("in", in_size);
  EnrollRepeatedOutputBn("out", out_size);
}

const PbMessage& TupleIdentityOp::GetCustomizedConf() const {
  return op_conf().tuple_identity_conf();
}

void TupleIdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  size_t bn_size = op_conf().tuple_identity_conf().in_size();
  FOR_RANGE(int, i, 0, bn_size) {
    *GetBlobDesc4BnInOp(output_bns().Get(i)) = *GetBlobDesc4BnInOp(input_bns().Get(i));
  }
}

void TupleIdentityOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new TupleIdentitySbpSignatureRule(this));
}

void TupleIdentityOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  FOR_RANGE(int32_t, i, 0, input_bns().size()) {
    *HasBatchDim4BnInOp(output_bns().Get(i)) = *HasBatchDim4BnInOp(input_bns().Get(i));
  }
}

REGISTER_OP(OperatorConf::kTupleIdentityConf, TupleIdentityOp);

}  // namespace oneflow
