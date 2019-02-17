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

class ParallelCastOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ParallelCastOpParallelSignature);
  ~ParallelCastOpParallelSignature() override = default;

  ParallelCastOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": A -> A"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    const auto& configured_sbp_parallel = GetSbpParallel(op().op_conf().parallel_cast_conf());
    if (SbpInferHint4BnInOp("in").sbp_parallel() == configured_sbp_parallel
        && parallel_ctx->parallel_num() != SbpInferHint4BnInOp("in").parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_ctx->parallel_num(),
                                                 SbpInferHint4BnInOp("in").parallel_num());
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    const auto& sbp_parallel = GetSbpParallel(op().op_conf().parallel_cast_conf());
    (*bn2sbp)["in"] = sbp_parallel;
    (*bn2sbp)["out"] = sbp_parallel;
  }
};

}  // namespace

void ParallelCastOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new ParallelCastOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kParallelCastConf, ParallelCastOp);

}  // namespace oneflow
