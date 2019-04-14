#include "oneflow/core/operator/reduce_sbp_signature_rule.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

class ReduceSplitSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSplitSignatureRule);
  ~ReduceSplitSignatureRule() override = default;

  ReduceSplitSignatureRule(const Operator* op, const HashSet<int64_t>& reduced_axes)
      : ParallelSbpSignatureRule(op), reduced_axes_(reduced_axes) {}

  const std::string Description() const override { return op().op_name() + ": (S, ...) -> S|P"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& data_ibn_hint = GetIbnHint(SbpInferHint4Ibn);
    if (data_ibn_hint.sbp_parallel().has_split_parallel() == false) {
      return MakeSbpSigMatchSignatureMismatch();
    } else {
      return MakeSbpSigMatchSuccess();
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    const auto& data_ibn_hint = GetIbnHint(SbpInferHint4Ibn);
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : op().input_bns()) { (*bn2sbp)[ibn] = data_ibn_hint.sbp_parallel(); }
    if (IsReduceAxisSplitted(SbpInferHint4Ibn)) {
      for (const auto& obn : op().output_bns()) { (*bn2sbp)[obn].mutable_partial_sum_parallel(); }
    } else {
      for (const auto& obn : op().output_bns()) { (*bn2sbp)[obn] = data_ibn_hint.sbp_parallel(); }
    }
  }

 private:
  bool IsReduceAxisSplitted(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn) const {
    const auto& data_ibn_hint = GetIbnHint(SbpInferHint4Ibn);
    if (data_ibn_hint.sbp_parallel().has_split_parallel() == false) { return false; }
    if (reduced_axes_.empty()) { return true; }
    return reduced_axes_.find(data_ibn_hint.sbp_parallel().split_parallel().axis())
           != reduced_axes_.end();
  }
  const SbpInferHint& GetIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn) const {
    const SbpInferHint* ret = nullptr;
    for (const auto& ibn : op().input_bns()) {
      if (ret != nullptr) {
        CHECK(SbpInferHint4Ibn(ibn).sbp_parallel() == ret->sbp_parallel());
      } else {
        ret = &SbpInferHint4Ibn(ibn);
      }
    }
    CHECK_NOTNULL(ret);
    return *ret;
  }

  HashSet<int64_t> reduced_axes_;
};

}  // namespace

std::unique_ptr<const SbpSignatureRule> MakeReduceSplitSignatureRule(
    const Operator* op, const HashSet<int64_t>& reduced_axes) {
  return std::make_unique<ReduceSplitSignatureRule>(op, reduced_axes);
}

void GetReduceSbpSignatureRules(const Operator* op, const HashSet<int64_t>& reduced_axes,
                                std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) {
  rules->emplace_back(MakeReduceSplitSignatureRule(op, reduced_axes));
  rules->emplace_back(MakeBroadcastSbpSignatureRule(op));
}

}  // namespace oneflow
