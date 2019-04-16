#include "oneflow/core/operator/broadcast_like_op.h"
#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

namespace {

class ReduceGradSplitSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGradSplitSignatureRule);
  ~ReduceGradSplitSignatureRule() override = default;

  ReduceGradSplitSignatureRule(const Operator* op, const std::string& like_ibn,
                               const HashSet<int64_t>& reduced_axes)
      : ParallelSbpSignatureRule(op), like_ibn_(like_ibn), reduced_axes_(reduced_axes) {}

  const std::string Description() const override { return op().op_name() + ": (B|S, ...) -> S"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& like_ibn_hint = SbpInferHint4Ibn(like_ibn_);
    if (like_ibn_hint.sbp_parallel().has_split_parallel() == false) {
      return MakeSbpSigMatchSignatureMismatch();
    } else {
      return MakeSbpSigMatchSuccess();
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    const auto& like_ibn_hint = SbpInferHint4Ibn(like_ibn_);
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : op().input_bns()) {
      if (ibn != like_ibn_
          && ReduceSbpUtil::IsReduceAxisSplitted(SbpInferHint4Ibn(like_ibn_), reduced_axes_)) {
        (*bn2sbp)[ibn].mutable_broadcast_parallel();
      } else {
        (*bn2sbp)[ibn] = like_ibn_hint.sbp_parallel();
      }
    }
    for (const auto& obn : op().output_bns()) { (*bn2sbp)[obn] = like_ibn_hint.sbp_parallel(); }
  }

 private:
  std::string like_ibn_;
  HashSet<int64_t> reduced_axes_;
};

void GetReduceGradSbpSignatureRules(const Operator* op, const std::string& like_ibn,
                                    const HashSet<int64_t>& reduced_axes,
                                    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) {
  rules->emplace_back(new ReduceGradSplitSignatureRule(op, like_ibn, reduced_axes));
  rules->emplace_back(MakeMultiIbnsBroadcastSbpSignatureRule(op));
}

}  // namespace

void BroadcastLikeOp::InitFromOpConf() {
  EnrollInputBn("x");
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollOutputBn("y");
}

const PbMessage& BroadcastLikeOp::GetCustomizedConf() const {
  return op_conf().broadcast_like_conf();
}

void BroadcastLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
}

void BroadcastLikeOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  const auto& reduced_axes = op_conf().broadcast_like_conf().reduced_axis();
  GetReduceGradSbpSignatureRules(this, "like", {reduced_axes.begin(), reduced_axes.end()}, rules);
}

REGISTER_OP(OperatorConf::kBroadcastLikeConf, BroadcastLikeOp);

}  // namespace oneflow
