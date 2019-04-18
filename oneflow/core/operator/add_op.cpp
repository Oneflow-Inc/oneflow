#include "oneflow/core/operator/add_op.h"
#include "oneflow/core/job/sbp_signature_rule.h"

namespace oneflow {

bool IsAllInputPartialSumParallel(
    const Operator& op,
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn) {
  for (const auto& ibn : op.input_bns()) {
    if (SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel() == false) { return false; }
  }
  return true;
}

bool IsAllInputBroadcastParallel(
    const Operator& op,
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn) {
  for (const auto& ibn : op.input_bns()) {
    if (SbpInferHint4Ibn(ibn).sbp_parallel().has_broadcast_parallel() == false) { return false; }
  }
  return true;
}

namespace {

class AddOpDataSplitSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddOpDataSplitSignatureRule);
  ~AddOpDataSplitSignatureRule() override = default;

  AddOpDataSplitSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (S(0), ...) -> S(0)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (IsAllInputPartialSumParallel(op(), SbpInferHint4Ibn)) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    if (IsAllInputBroadcastParallel(op(), SbpInferHint4Ibn)) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : op().input_bns()) {
      (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0);
    }
    for (const auto& obn : op().output_bns()) {
      (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0);
    }
  }
};

}  // namespace

void AddOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_conf()); }
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }
void AddOp::VirtualFixInDiffBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  if (!Global<JobDesc>::Get()->enable_blob_mem_sharing()) { return; }
  int64_t blob_mem_id = oneflow_cast<int64_t>(NewUniqueId());
  FOR_RANGE(size_t, i, 0, input_diff_bns().size()) {
    GetBlobDesc4BnInOp(input_diff_bns().Get(i))->set_blob_mem_id(blob_mem_id);
  }
}

void AddOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new AddOpDataSplitSignatureRule(this));
  rules->emplace_back(MakeMultiIbnsBroadcastSbpSignatureRule(this));
  rules->emplace_back(MakePartialSumSignatureRule(this));
}

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
