#include "oneflow/core/operator/parallel_cast_op.h"

namespace oneflow {

namespace {

SbpParallel GetSbpParallel(const ParallelCastOpConf& conf) {
  SbpParallel ret;
  if (conf.has_split_parallel()) {
    *ret.mutable_split_parallel() = conf.split_parallel();
  } else if (conf.has_broadcast_parallel()) {
    *ret.mutable_broadcast_parallel() = conf.broadcast_parallel();
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

class ParallelCastSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ParallelCastSbpSignatureRule);
  ~ParallelCastSbpSignatureRule() override = default;

  ParallelCastSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": A -> A"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const auto& configured_sbp_parallel = GetSbpParallel(op().op_conf().parallel_cast_conf());
    if (SbpInferHint4BnInOp("in").sbp_parallel() == configured_sbp_parallel
        && parallel_desc.parallel_num() != SbpInferHint4BnInOp("in").parallel_num()) {
      return MakeSbpSigMatchParallelNumError(parallel_desc.parallel_num(),
                                             SbpInferHint4BnInOp("in").parallel_num());
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const auto& sbp_parallel = GetSbpParallel(op().op_conf().parallel_cast_conf());
    (*bn2sbp)["in"] = sbp_parallel;
    (*bn2sbp)["out"] = sbp_parallel;
  }
};

}  // namespace

void ParallelCastOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new ParallelCastSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kParallelCastConf, ParallelCastOp);

}  // namespace oneflow
