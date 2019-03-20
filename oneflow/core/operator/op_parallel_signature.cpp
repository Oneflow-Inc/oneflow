#include "oneflow/core/operator/op_parallel_signature.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

const OpParallelMatchResult MakeOpParallelMatchSuccess() {
  OpParallelMatchResult success;
  success.mutable_success();
  return success;
}

const OpParallelMatchResult MakeOpParallelMatchSignatureMismatch() {
  OpParallelMatchResult signature_mismatch;
  signature_mismatch.mutable_fail()->mutable_signature_mismatch();
  return signature_mismatch;
}

const OpParallelMatchResult MakeOpParallelMatchParallelPolicyError(ParallelPolicy configured,
                                                                   ParallelPolicy expected) {
  OpParallelMatchResult policy_error;
  auto* err = policy_error.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
  err->set_configured(configured);
  err->set_expected(expected);
  return policy_error;
}

const OpParallelMatchResult MakeOpParallelMatchParallelNumError(int64_t configured,
                                                                int64_t expected) {
  OpParallelMatchResult parallel_num_error;
  auto* err = parallel_num_error.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
  err->set_configured(configured);
  err->set_expected(expected);
  return parallel_num_error;
}

const OpParallelMatchResult MakeOpParallelMatchDeviceSetError(const std::string& configured,
                                                              const std::string& expected) {
  OpParallelMatchResult parallel_num_error;
  auto* err = parallel_num_error.mutable_fail()->mutable_conf_error()->mutable_device_set_error();
  err->set_configured(configured);
  err->set_expected(expected);
  return parallel_num_error;
}

namespace {

class DataSplitOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataSplitOpParallelSignature);
  ~DataSplitOpParallelSignature() override = default;

  DataSplitOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (S(0), ...) -> (S(0), ...)";
  }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    bool is_data_split = true;
    auto IsDataSplit = [&](const SbpInferHint& sbp_infer_hint) {
      return !sbp_infer_hint.sbp_parallel().has_broadcast_parallel()
             || sbp_infer_hint.parallel_num() == 1;
    };
    for (const auto& bn : op().input_bns()) {
      const SbpInferHint& sbp_infer_hint = SbpInferHint4Ibn(bn);
      if (!IsDataSplit(sbp_infer_hint)) {
        is_data_split = false;
        break;
      }
    }
    if (!is_data_split) { return MakeOpParallelMatchSignatureMismatch(); }
    if (parallel_desc.policy() == kDataParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
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

class BroadcastOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastOpParallelSignature);
  ~BroadcastOpParallelSignature() override = default;

  BroadcastOpParallelSignature(const Operator* op) : OpParallelSignature(op) {
    CHECK_EQ(op->input_bns().size(), 1);
    CHECK(op->model_bns().empty());
    CHECK(op->const_model_bns().empty());
  }

  const std::string Description() const override { return op().op_name() + ": (B,) -> (B, ...)"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (!SbpInferHint4Ibn(op().SoleIbn()).sbp_parallel().has_broadcast_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    int64_t expected_parallel_num = SbpInferHint4Ibn(op().SoleIbn()).parallel_num();
    bool parallel_policy_matched = (parallel_desc.policy() == kDataParallel);
    bool parallel_num_matched =
        (parallel_desc.parallel_num() == expected_parallel_num && parallel_desc.parallel_num() > 1);
    if (parallel_policy_matched && parallel_num_matched) {
      return MakeOpParallelMatchSuccess();
    } else {
      OpParallelMatchResult ret;
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

class DS_MB_2_DS_OpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DS_MB_2_DS_OpParallelSignature);
  ~DS_MB_2_DS_OpParallelSignature() override = default;

  DS_MB_2_DS_OpParallelSignature(const Operator* op) : OpParallelSignature(op) {
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

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& model_sbp_infer_hint = SbpInferHint4Ibn(model_input_bns_.at(0));
    for (const auto& bn : model_input_bns_) {
      if (model_sbp_infer_hint.parallel_desc() != SbpInferHint4Ibn(bn).parallel_desc()) {
        return MakeOpParallelMatchSignatureMismatch();
      }
      if (!SbpInferHint4Ibn(bn).sbp_parallel().has_broadcast_parallel()) {
        return MakeOpParallelMatchSignatureMismatch();
      }
    }
    bool parallel_policy_matched = (parallel_desc.policy() == kDataParallel);
    bool parallel_num_matched = (parallel_desc.parallel_num() == model_sbp_infer_hint.parallel_num()
                                 && parallel_desc.parallel_num() > 1);
    if (!parallel_policy_matched || !parallel_num_matched) {
      OpParallelMatchResult ret;
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
    return MakeOpParallelMatchSuccess();
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

class SoleIbnOpModelSplitOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoleIbnOpModelSplitOpParallelSignature);
  ~SoleIbnOpModelSplitOpParallelSignature() override = default;

  SoleIbnOpModelSplitOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (S,) -> (S, ...)"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& sbp_infer_hint = SbpInferHint4Ibn(op().SoleIbn());
    if (!(sbp_infer_hint.is_model_split()
          || (sbp_infer_hint.is_data_split() && sbp_infer_hint.split_axis() > 0))) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    int64_t expected_parallel_num = sbp_infer_hint.parallel_num();
    bool parallel_policy_matched = (parallel_desc.policy() == kModelParallel);
    bool parallel_num_matched = (parallel_desc.parallel_num() == expected_parallel_num);
    if (!(parallel_policy_matched && parallel_num_matched)) {
      OpParallelMatchResult ret;
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
    return MakeOpParallelMatchSuccess();
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

class ModelBnOpModelSplitOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelBnOpModelSplitOpParallelSignature);
  ~ModelBnOpModelSplitOpParallelSignature() override = default;

  ModelBnOpModelSplitOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (B, ...) -> (S, ...)";
  }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
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

class DB_MS_2_MS_OpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DB_MS_2_MS_OpParallelSignature);
  ~DB_MS_2_MS_OpParallelSignature() override = default;

  DB_MS_2_MS_OpParallelSignature(const Operator* op, std::function<bool(int32_t)> IsExpectedAxis)
      : OpParallelSignature(op), IsExpectedAxis_(IsExpectedAxis) {
    for (const auto& bn : op->input_bns()) {
      if (op->IsInputBlobAllowedModelSplit(bn)) {
        model_input_bns_.push_back(bn);
      } else {
        data_input_bns_.push_back(bn);
      }
    }
    CHECK_GT(data_input_bns_.size(), 0);
<<<<<<< HEAD
    CHECK_EQ(model_input_bns_.size(), 1);
=======
    CHECK_GT(model_input_bns_.size(), 0);
>>>>>>> origin/dev_multi_model_ibn_sbp_signature
  }

  const std::string Description() const override {
    return op().op_name() + ": (B, S, ...) -> (S, ...)";
  }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& model_sbp_infer_hint = SbpInferHint4Ibn(model_input_bns_.at(0));
    for (const auto& bn : model_input_bns_) {
      if (model_sbp_infer_hint.parallel_desc() != SbpInferHint4Ibn(bn).parallel_desc()) {
        return MakeOpParallelMatchSignatureMismatch();
      }
      if (model_sbp_infer_hint.sbp_parallel() != SbpInferHint4Ibn(bn).sbp_parallel()) {
        return MakeOpParallelMatchSignatureMismatch();
      }
    }
    if (!(model_sbp_infer_hint.is_model_split()
          && IsValidSplit(model_sbp_infer_hint.split_axis()))) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (parallel_desc.policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
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

}  // namespace

std::unique_ptr<const OpParallelSignature> MakeDataSplitOpParallelSignature(const Operator* op) {
  return std::unique_ptr<const OpParallelSignature>(new DataSplitOpParallelSignature(op));
}

std::unique_ptr<const OpParallelSignature> MakeBroadcastOpParallelSignature(const Operator* op) {
  return std::unique_ptr<const OpParallelSignature>(new BroadcastOpParallelSignature(op));
}

std::unique_ptr<const OpParallelSignature> MakeModelSplitOpParallelSignature(const Operator* op) {
  if (op->IsSoleInputBlobAllowedModelSplit()) {
    return std::unique_ptr<const OpParallelSignature>(
        new SoleIbnOpModelSplitOpParallelSignature(op));
  } else {
    CHECK(!op->model_bns().empty() || !op->const_model_bns().empty());
    return std::unique_ptr<const OpParallelSignature>(
        new ModelBnOpModelSplitOpParallelSignature(op));
  }
}

std::unique_ptr<const OpParallelSignature> Make_DS_MB_2_DS_OpParallelSignature(const Operator* op) {
  return std::unique_ptr<const OpParallelSignature>(new DS_MB_2_DS_OpParallelSignature(op));
}

std::unique_ptr<const OpParallelSignature> Make_DB_MS_2_MS_OpParallelSignature(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis) {
  return std::unique_ptr<const OpParallelSignature>(
      new DB_MS_2_MS_OpParallelSignature(op, IsExpectedAxis));
}

}  // namespace oneflow
