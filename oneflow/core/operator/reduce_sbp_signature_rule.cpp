#include "oneflow/core/operator/reduce_sbp_signature_rule.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

class ReduceSplitSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSplitSignatureRule);
  ~ReduceSplitSignatureRule() override = default;

  ReduceSplitSignatureRule(const Operator* op, const std::string& data_ibn,
                           const HashSet<int64_t>& reduced_axes)
      : ParallelSbpSignatureRule(op), data_ibn_(data_ibn), reduced_axes_(reduced_axes) {}

  const std::string Description() const override { return op().op_name() + ": (S, ...) -> S|P"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& data_ibn_hint = SbpInferHint4Ibn(data_ibn_);
    if (data_ibn_hint.sbp_parallel().has_split_parallel() == false) {
      return MakeSbpSigMatchSignatureMismatch();
    } else {
      return MakeSbpSigMatchSuccess();
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    const auto& data_ibn_hint = SbpInferHint4Ibn(data_ibn_);
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : op().input_bns()) {
      (*bn2sbp)[ibn] = SbpInferHint4Ibn(ibn).sbp_parallel();
    }
    if (IsReduceAxisSplitted(SbpInferHint4Ibn)) {
      for (const auto& obn : op().output_bns()) { (*bn2sbp)[obn].mutable_partial_sum_parallel(); }
    } else {
      for (const auto& obn : op().output_bns()) { (*bn2sbp)[obn] = data_ibn_hint.sbp_parallel(); }
    }
  }

 private:
  bool IsReduceAxisSplitted(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn) const {
    const auto& data_ibn_hint = SbpInferHint4Ibn(data_ibn_);
    if (data_ibn_hint.sbp_parallel().has_split_parallel() == false) { return false; }
    if (reduced_axes_.empty()) { return true; }
    return reduced_axes_.find(data_ibn_hint.sbp_parallel().split_parallel().axis())
           != reduced_axes_.end();
  }

  std::string data_ibn_;
  HashSet<int64_t> reduced_axes_;
};

}  // namespace

std::unique_ptr<const SbpSignatureRule> MakeReduceSplitSignatureRule(
    const Operator* op, const std::string& data_ibn, const HashSet<int64_t>& reduced_axes) {
  return std::make_unique<ReduceSplitSignatureRule>(op, data_ibn, reduced_axes);
}

void GetReduceSbpSignatureRules(const Operator* op, const std::string& data_ibn,
                                const HashSet<int64_t>& reduced_axes,
                                std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) {
  rules->emplace_back(MakeReduceSplitSignatureRule(op, data_ibn, reduced_axes));
  rules->emplace_back(MakeBroadcastSbpSignatureRule(op));
}

}  // namespace oneflow
