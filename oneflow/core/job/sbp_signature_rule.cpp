#include "oneflow/core/job/sbp_signature_rule.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

const SbpSigMatchResult MakeSbpSigMatchSuccess() {
  SbpSigMatchResult success;
  success.mutable_success();
  return success;
}

const SbpSigMatchResult MakeSbpSigMatchSignatureMismatch() {
  SbpSigMatchResult signature_mismatch;
  signature_mismatch.mutable_fail()->mutable_signature_mismatch();
  return signature_mismatch;
}

const SbpSigMatchResult MakeSbpSigMatchParallelPolicyError(ParallelPolicy configured,
                                                           ParallelPolicy expected) {
  SbpSigMatchResult policy_error;
  auto* err = policy_error.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
  err->set_configured(configured);
  err->set_expected(expected);
  return policy_error;
}

const SbpSigMatchResult MakeSbpSigMatchParallelNumError(int64_t configured, int64_t expected) {
  SbpSigMatchResult parallel_num_error;
  auto* err = parallel_num_error.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
  err->set_configured(configured);
  err->set_expected(expected);
  return parallel_num_error;
}

const SbpSigMatchResult MakeSbpSigMatchDeviceSetError(const std::string& configured,
                                                      const std::string& expected) {
  SbpSigMatchResult parallel_num_error;
  auto* err = parallel_num_error.mutable_fail()->mutable_conf_error()->mutable_device_set_error();
  err->set_configured(configured);
  err->set_expected(expected);
  return parallel_num_error;
}

const SbpSigMatchResult SbpSignatureRule::MatchIf(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const SbpSignature& conf_obn_sbp_sig_hint, const ParallelDesc& parallel_desc) const {
  if (conf_obn_sbp_sig_hint.bn_in_op2sbp_parallel().size() > 0) {
    const auto& result = MatchByObnSbpSigHint(SbpInferHint4Ibn, conf_obn_sbp_sig_hint);
    if (result.has_success()) { return result; }
  }
  return MatchByIbnHintIf(SbpInferHint4Ibn, parallel_desc);
}

const SbpSigMatchResult ParallelSbpSignatureRule::MatchByIbnHintIf(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  if (parallel_desc.parallel_num() == 1) {
    return MakeSbpSigMatchSignatureMismatch();
  } else if (parallel_desc.parallel_num() > 1) {
    return MatchByIbnHint(SbpInferHint4Ibn, parallel_desc);
  } else {
    UNIMPLEMENTED();
  }
}

const SbpSigMatchResult ParallelSbpSignatureRule::MatchByObnSbpSigHint(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const SbpSignature& conf_obn_sbp_sig_hint) const {
  SbpSignature generated_sbp_signature;
  GenerateSignature(SbpInferHint4Ibn, &generated_sbp_signature);
  auto& bn2sbp = generated_sbp_signature.bn_in_op2sbp_parallel();
  for (const auto& pair : conf_obn_sbp_sig_hint.bn_in_op2sbp_parallel()) {
    CHECK(bn2sbp.find(pair.first) != bn2sbp.end());
    if (bn2sbp.at(pair.first) != pair.second) { return MakeSbpSigMatchSignatureMismatch(); }
  }
  return MakeSbpSigMatchSuccess();
}

namespace {

class UnparallelSbpSignatureRule final : public SbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnparallelSbpSignatureRule);
  ~UnparallelSbpSignatureRule() override = default;

  UnparallelSbpSignatureRule(const Operator* op) : SbpSignatureRule(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (U, ...) -> (U, ...)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  const SbpSigMatchResult MatchByIbnHintIf(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.parallel_num() == 1) {
      return MatchByIbnHint(SbpInferHint4Ibn, parallel_desc);
    } else if (parallel_desc.parallel_num() > 1) {
      return MakeSbpSigMatchSignatureMismatch();
    } else {
      UNIMPLEMENTED();
    }
  }

  const SbpSigMatchResult MatchByObnSbpSigHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const SbpSignature& conf_obn_sbp_sig_hint) const {
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

class DataSplitSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataSplitSbpSignatureRule);
  ~DataSplitSbpSignatureRule() override = default;

  DataSplitSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (S(0), ...) -> (S(0), ...)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    bool is_data_split = true;
    auto IsDataSplit = [&](const SbpInferHint& sbp_infer_hint) {
      return !sbp_infer_hint.sbp_parallel().has_broadcast_parallel();
    };
    for (const auto& bn : op().input_bns()) {
      const SbpInferHint& sbp_infer_hint = SbpInferHint4Ibn(bn);
      if (!IsDataSplit(sbp_infer_hint)) {
        is_data_split = false;
        break;
      }
    }
    if (!is_data_split) { return MakeSbpSigMatchSignatureMismatch(); }
    if (parallel_desc.policy() == kDataParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
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

class BroadcastSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastSbpSignatureRule);
  ~BroadcastSbpSignatureRule() override = default;

  BroadcastSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {
    CHECK_EQ(op->input_bns().size(), 1);
    CHECK(op->model_bns().empty());
    CHECK(op->const_model_bns().empty());
  }

  const std::string Description() const override { return op().op_name() + ": (B,) -> (B, ...)"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (!SbpInferHint4Ibn(op().SoleIbn()).sbp_parallel().has_broadcast_parallel()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    int64_t expected_parallel_num = SbpInferHint4Ibn(op().SoleIbn()).parallel_num();
    bool parallel_policy_matched = (parallel_desc.policy() == kDataParallel);
    bool parallel_num_matched = parallel_desc.parallel_num() == expected_parallel_num;
    if (parallel_policy_matched && parallel_num_matched) {
      return MakeSbpSigMatchSuccess();
    } else {
      SbpSigMatchResult ret;
      if (!parallel_policy_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
        err->set_configured(parallel_desc.policy());
        err->set_expected(kDataParallel);
      } else {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
        err->set_configured(parallel_desc.parallel_num());
        err->set_expected(expected_parallel_num);
      }
      return ret;
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& bn : op().input_bns()) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
    for (const auto& bn : op().output_bns()) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
  }
};

class DS_MB_2_DS_SbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DS_MB_2_DS_SbpSignatureRule);
  ~DS_MB_2_DS_SbpSignatureRule() override = default;

  DS_MB_2_DS_SbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {
    for (const auto& bn : op->input_bns()) {
      if (op->IsInputBlobAllowedModelSplit(bn)) {
        model_input_bns_.push_back(bn);
      } else {
        data_input_bns_.push_back(bn);
      }
    }
    CHECK_GT(data_input_bns_.size(), 0);
    CHECK_GT(model_input_bns_.size(), 0);
  }

  const std::string Description() const override {
    return op().op_name() + ": (B, S(0), ...) -> (S(0), ...)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& model_sbp_infer_hint = SbpInferHint4Ibn(model_input_bns_.at(0));
    for (const auto& bn : model_input_bns_) {
      if (model_sbp_infer_hint.parallel_desc() != SbpInferHint4Ibn(bn).parallel_desc()) {
        return MakeSbpSigMatchSignatureMismatch();
      }
      if (!SbpInferHint4Ibn(bn).sbp_parallel().has_broadcast_parallel()) {
        return MakeSbpSigMatchSignatureMismatch();
      }
    }
    bool parallel_policy_matched = (parallel_desc.policy() == kDataParallel);
    bool parallel_num_matched = parallel_desc.parallel_num() == model_sbp_infer_hint.parallel_num();
    if (!parallel_policy_matched || !parallel_num_matched) {
      SbpSigMatchResult ret;
      if (!parallel_policy_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
        err->set_configured(parallel_desc.policy());
        err->set_expected(kDataParallel);
      }
      if (!parallel_num_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
        err->set_configured(parallel_desc.parallel_num());
        err->set_expected(model_sbp_infer_hint.parallel_num());
      }
      return ret;
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& bn : data_input_bns_) { (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0); }
    for (const auto& bn : model_input_bns_) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
    }
  }

 private:
  std::vector<std::string> data_input_bns_;
  std::vector<std::string> model_input_bns_;
};

class SoleIbnOpModelSplitSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoleIbnOpModelSplitSbpSignatureRule);
  ~SoleIbnOpModelSplitSbpSignatureRule() override = default;

  SoleIbnOpModelSplitSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (S,) -> (S, ...)"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& sbp_infer_hint = SbpInferHint4Ibn(op().SoleIbn());
    if (!(sbp_infer_hint.is_model_split()
          || (sbp_infer_hint.is_data_split() && sbp_infer_hint.split_axis() > 0))) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    int64_t expected_parallel_num = sbp_infer_hint.parallel_num();
    bool parallel_policy_matched = (parallel_desc.policy() == kModelParallel);
    bool parallel_num_matched = (parallel_desc.parallel_num() == expected_parallel_num);
    if (!(parallel_policy_matched && parallel_num_matched)) {
      SbpSigMatchResult ret;
      if (!parallel_policy_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
        err->set_configured(parallel_desc.policy());
        err->set_expected(kModelParallel);
      }
      if (!parallel_num_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
        err->set_configured(parallel_desc.parallel_num());
        err->set_expected(parallel_num_matched);
      }
      return ret;
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)[op().SoleIbn()] = SbpInferHint4Ibn(op().SoleIbn()).sbp_parallel();
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(
          op().OutputBlobModelSplitAxis(SbpInferHint4Ibn, bn));
    }
  }
};

class ModelBnOpModelSplitSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelBnOpModelSplitSbpSignatureRule);
  ~ModelBnOpModelSplitSbpSignatureRule() override = default;

  ModelBnOpModelSplitSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (B, ...) -> (S, ...)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kModelParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& bn : op().input_bns()) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(
          op().OutputBlobModelSplitAxis(SbpInferHint4Ibn, bn));
    }
  }
};

class DB_MS_2_MS_SbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DB_MS_2_MS_SbpSignatureRule);
  ~DB_MS_2_MS_SbpSignatureRule() override = default;

  DB_MS_2_MS_SbpSignatureRule(const Operator* op, std::function<bool(int32_t)> IsExpectedAxis)
      : ParallelSbpSignatureRule(op), IsExpectedAxis_(IsExpectedAxis) {
    for (const auto& bn : op->input_bns()) {
      if (op->IsInputBlobAllowedModelSplit(bn)) {
        model_input_bns_.push_back(bn);
      } else {
        data_input_bns_.push_back(bn);
      }
    }
    CHECK_GT(data_input_bns_.size(), 0);
    CHECK_GT(model_input_bns_.size(), 0);
  }

  const std::string Description() const override {
    return op().op_name() + ": (B, S, ...) -> (S, ...)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& model_sbp_infer_hint = SbpInferHint4Ibn(model_input_bns_.at(0));
    for (const auto& bn : model_input_bns_) {
      if (model_sbp_infer_hint.parallel_desc() != SbpInferHint4Ibn(bn).parallel_desc()) {
        return MakeSbpSigMatchSignatureMismatch();
      }
      if (model_sbp_infer_hint.sbp_parallel() != SbpInferHint4Ibn(bn).sbp_parallel()) {
        return MakeSbpSigMatchSignatureMismatch();
      }
    }
    if (!(model_sbp_infer_hint.is_model_split()
          && IsValidSplit(model_sbp_infer_hint.split_axis()))) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    if (parallel_desc.policy() == kModelParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    for (const auto& bn : data_input_bns_) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
    for (const auto& bn : model_input_bns_) { (*bn2sbp)[bn] = SbpInferHint4Ibn(bn).sbp_parallel(); }
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(
          op().OutputBlobModelSplitAxis(SbpInferHint4Ibn, bn));
    }
  }

 private:
  bool IsValidSplit(int32_t axis) const { return axis != -1 && IsExpectedAxis_(axis); }

  const std::function<bool(int32_t)> IsExpectedAxis_;
  std::vector<std::string> data_input_bns_;
  std::vector<std::string> model_input_bns_;
};

}  // namespace

std::unique_ptr<const SbpSignatureRule> MakeUnparallelSbpSignatureRule(const Operator* op) {
  return std::unique_ptr<const SbpSignatureRule>(new UnparallelSbpSignatureRule(op));
}

std::unique_ptr<const SbpSignatureRule> MakeDataSplitSbpSignatureRule(const Operator* op) {
  return std::unique_ptr<const SbpSignatureRule>(new DataSplitSbpSignatureRule(op));
}

std::unique_ptr<const SbpSignatureRule> MakeBroadcastSbpSignatureRule(const Operator* op) {
  return std::unique_ptr<const SbpSignatureRule>(new BroadcastSbpSignatureRule(op));
}

std::unique_ptr<const SbpSignatureRule> MakeModelSplitSbpSignatureRule(const Operator* op) {
  if (op->IsSoleInputBlobAllowedModelSplit()) {
    return std::unique_ptr<const SbpSignatureRule>(new SoleIbnOpModelSplitSbpSignatureRule(op));
  } else {
    CHECK(!op->model_bns().empty() || !op->const_model_bns().empty());
    return std::unique_ptr<const SbpSignatureRule>(new ModelBnOpModelSplitSbpSignatureRule(op));
  }
}

std::unique_ptr<const SbpSignatureRule> Make_DS_MB_2_DS_SbpSignatureRule(const Operator* op) {
  return std::unique_ptr<const SbpSignatureRule>(new DS_MB_2_DS_SbpSignatureRule(op));
}

std::unique_ptr<const SbpSignatureRule> Make_DB_MS_2_MS_SbpSignatureRule(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis) {
  return std::unique_ptr<const SbpSignatureRule>(
      new DB_MS_2_MS_SbpSignatureRule(op, IsExpectedAxis));
}

}  // namespace oneflow
