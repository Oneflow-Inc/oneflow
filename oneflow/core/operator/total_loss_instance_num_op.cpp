#include "oneflow/core/operator/total_loss_instance_num_op.h"

namespace oneflow {

namespace {

class TotalLossInstanceSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TotalLossInstanceSbpSignatureRule);
  ~TotalLossInstanceSbpSignatureRule() override = default;

  TotalLossInstanceSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (U, ...) -> (U,)"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    for (const std::string& ibn : op().input_bns()) {
      if (!SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel()) {
        return MakeSbpSigMatchSignatureMismatch();
      }
    }
    if (parallel_desc.parallel_num() != 1) {
      return MakeSbpSigMatchParallelNumError(parallel_desc.parallel_num(), 1);
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& bn : op().input_bns()) { (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0); }
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
    }
  }
};

}  // namespace

void TotalLossInstanceNumOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_total_loss_instance_num_conf());
}

void TotalLossInstanceNumOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  for (const std::string& ibn : input_bns()) {
    CHECK(*GetBlobDesc4BnInOp(ibn) == *GetBlobDesc4BnInOp(input_bns().Get(0)));
  }
}

const PbMessage& TotalLossInstanceNumOp::GetCustomizedConf() const {
  return op_conf().total_loss_instance_num_conf();
}

void TotalLossInstanceNumOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new TotalLossInstanceSbpSignatureRule(this));
}

REGISTER_CPU_OP(OperatorConf::kTotalLossInstanceNumConf, TotalLossInstanceNumOp);

}  // namespace oneflow
