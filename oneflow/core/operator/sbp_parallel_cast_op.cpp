#include "oneflow/core/operator/sbp_parallel_cast_op.h"

namespace oneflow {

namespace {

class SbpParallelCastOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SbpParallelCastOpParallelSignature);
  ~SbpParallelCastOpParallelSignature() override = default;

  SbpParallelCastOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": A -> A"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    const auto& sbp_parallel_conf = op().op_conf().sbp_parallel_cast_conf().sbp();
    if (SbpInferHint4BnInOp("in").sbp_parallel() == sbp_parallel_conf
        && parallel_ctx->parallel_num() != SbpInferHint4BnInOp("in").parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_ctx->parallel_num(),
                                                 SbpInferHint4BnInOp("in").parallel_num());
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    const auto& sbp_parallel_conf = op().op_conf().sbp_parallel_cast_conf().sbp();
    (*bn2sbp)["in"] = sbp_parallel_conf;
    (*bn2sbp)["out"] = sbp_parallel_conf;
  }
};

}  // namespace

void SbpParallelCastOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new SbpParallelCastOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kSbpParallelCastConf, SbpParallelCastOp);

}  // namespace oneflow
