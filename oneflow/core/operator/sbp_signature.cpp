#include "oneflow/core/operator/sbp_signature.h"
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

const SbpSigMatchResult ParallelSbpSignature::GetMatchResultIf(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  if (parallel_desc.parallel_num() == 1) {
    return MakeSbpSigMatchSignatureMismatch();
  } else if (parallel_desc.parallel_num() > 1) {
    return GetMatchResult(SbpInferHint4Ibn, parallel_desc);
  } else {
    UNIMPLEMENTED();
  }
}

namespace {

class UnparallelSbpSignature final : public SbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnparallelSbpSignature);
  ~UnparallelSbpSignature() override = default;

  UnparallelSbpSignature(const Operator* op) : SbpSignature(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (U, ...) -> (U, ...)";
  }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  const SbpSigMatchResult GetMatchResultIf(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.parallel_num() == 1) {
      return GetMatchResult(SbpInferHint4Ibn, parallel_desc);
    } else if (parallel_desc.parallel_num() > 1) {
      return MakeSbpSigMatchSignatureMismatch();
    } else {
      UNIMPLEMENTED();
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : op().input_bns()) { (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0); }
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
    }
  }
};

class DataSplitSbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataSplitSbpSignature);
  ~DataSplitSbpSignature() override = default;

  DataSplitSbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (S(0), ...) -> (S(0), ...)";
  }

  const SbpSigMatchResult GetMatchResult(
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
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : op().input_bns()) { (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0); }
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
    }
  }
};

class BroadcastSbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastSbpSignature);
  ~BroadcastSbpSignature() override = default;

  BroadcastSbpSignature(const Operator* op) : ParallelSbpSignature(op) {
    CHECK_EQ(op->input_bns().size(), 1);
    CHECK(op->model_bns().empty());
    CHECK(op->const_model_bns().empty());
  }

  const std::string Description() const override { return op().op_name() + ": (B,) -> (B, ...)"; }

  const SbpSigMatchResult GetMatchResult(
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
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : op().input_bns()) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
    for (const auto& bn : op().output_bns()) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
  }
};

class DS_MB_2_DS_SbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DS_MB_2_DS_SbpSignature);
  ~DS_MB_2_DS_SbpSignature() override = default;

  DS_MB_2_DS_SbpSignature(const Operator* op) : ParallelSbpSignature(op) {
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

  const SbpSigMatchResult GetMatchResult(
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
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
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

class SoleIbnOpModelSplitSbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoleIbnOpModelSplitSbpSignature);
  ~SoleIbnOpModelSplitSbpSignature() override = default;

  SoleIbnOpModelSplitSbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (S,) -> (S, ...)"; }

  const SbpSigMatchResult GetMatchResult(
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
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)[op().SoleIbn()] = SbpInferHint4Ibn(op().SoleIbn()).sbp_parallel();
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(
          op().OutputBlobModelSplitAxis(SbpInferHint4Ibn, bn));
    }
  }
};

class ModelBnOpModelSplitSbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelBnOpModelSplitSbpSignature);
  ~ModelBnOpModelSplitSbpSignature() override = default;

  ModelBnOpModelSplitSbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (B, ...) -> (S, ...)";
  }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kModelParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : op().input_bns()) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(
          op().OutputBlobModelSplitAxis(SbpInferHint4Ibn, bn));
    }
  }
};

class DB_MS_2_MS_SbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DB_MS_2_MS_SbpSignature);
  ~DB_MS_2_MS_SbpSignature() override = default;

  DB_MS_2_MS_SbpSignature(const Operator* op, std::function<bool(int32_t)> IsExpectedAxis)
      : ParallelSbpSignature(op), IsExpectedAxis_(IsExpectedAxis) {
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

  const SbpSigMatchResult GetMatchResult(
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
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
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

class IdentitySbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentitySbpSignature);
  ~IdentitySbpSignature() override = default;

  IdentitySbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (A,) -> (A,)"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    CHECK_EQ(op().input_bns().size(), 1);
    CHECK_EQ(op().output_bns().size(), 1);
    const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp(op().input_bns().Get(0));
    if (in_sbp_infer_hint.parallel_num() != parallel_desc.parallel_num()) {
      return MakeSbpSigMatchParallelNumError(parallel_desc.parallel_num(),
                                             in_sbp_infer_hint.parallel_num());
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    const SbpParallel& sbp_parallel = SbpInferHint4BnInOp(op().input_bns().Get(0)).sbp_parallel();
    (*bn2sbp)[op().input_bns().Get(0)] = sbp_parallel;
    (*bn2sbp)[op().output_bns().Get(0)] = sbp_parallel;
  }
};

}  // namespace

std::unique_ptr<const SbpSignature> MakeUnparallelSbpSignature(const Operator* op) {
  return std::unique_ptr<const SbpSignature>(new UnparallelSbpSignature(op));
}

std::unique_ptr<const SbpSignature> MakeDataSplitSbpSignature(const Operator* op) {
  return std::unique_ptr<const SbpSignature>(new DataSplitSbpSignature(op));
}

std::unique_ptr<const SbpSignature> MakeBroadcastSbpSignature(const Operator* op) {
  return std::unique_ptr<const SbpSignature>(new BroadcastSbpSignature(op));
}

std::unique_ptr<const SbpSignature> MakeModelSplitSbpSignature(const Operator* op) {
  if (op->IsSoleInputBlobAllowedModelSplit()) {
    return std::unique_ptr<const SbpSignature>(new SoleIbnOpModelSplitSbpSignature(op));
  } else {
    CHECK(!op->model_bns().empty() || !op->const_model_bns().empty());
    return std::unique_ptr<const SbpSignature>(new ModelBnOpModelSplitSbpSignature(op));
  }
}

std::unique_ptr<const SbpSignature> Make_DS_MB_2_DS_SbpSignature(const Operator* op) {
  return std::unique_ptr<const SbpSignature>(new DS_MB_2_DS_SbpSignature(op));
}

std::unique_ptr<const SbpSignature> Make_DB_MS_2_MS_SbpSignature(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis) {
  return std::unique_ptr<const SbpSignature>(new DB_MS_2_MS_SbpSignature(op, IsExpectedAxis));
}

std::unique_ptr<const SbpSignature> MakeIdentitySbpSignature(const Operator* op) {
  return std::unique_ptr<const SbpSignature>(new IdentitySbpSignature(op));
}

}  // namespace oneflow
