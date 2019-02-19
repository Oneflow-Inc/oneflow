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
      const ParallelContext* parallel_ctx) const override {
    bool is_data_split = true;
    for (const auto& bn : op().input_bns()) {
      const SbpInferHint& sbp_infer_hint = SbpInferHint4Ibn(bn);
      if (!sbp_infer_hint.is_data_blob()) {
        is_data_split = false;
        break;
      }
    }
    if (!is_data_split) { return MakeOpParallelMatchSignatureMismatch(); }
    if (parallel_ctx->policy() == kDataParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kDataParallel);
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
      const ParallelContext* parallel_ctx) const override {
    if (!SbpInferHint4Ibn(op().SoleIbn()).sbp_parallel().has_broadcast_parallel()) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    int64_t expected_parallel_num = SbpInferHint4Ibn(op().SoleIbn()).parallel_num();
    bool parallel_policy_matched = (parallel_ctx->policy() == kDataParallel);
    bool parallel_num_matched = (parallel_ctx->parallel_num() == expected_parallel_num);
    if (parallel_policy_matched && parallel_num_matched) {
      return MakeOpParallelMatchSuccess();
    } else {
      OpParallelMatchResult ret;
      if (!parallel_policy_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
        err->set_configured(parallel_ctx->policy());
        err->set_expected(kDataParallel);
      } else {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
        err->set_configured(parallel_ctx->parallel_num());
        err->set_expected(parallel_num_matched);
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
    std::vector<std::string> model_input_bns;
    for (const auto& bn : op->input_bns()) {
      if (op->IsInputBlobAllowedModelSplit(bn)) {
        model_input_bns.push_back(bn);
      } else {
        data_input_bns_.push_back(bn);
      }
    }
    CHECK_GT(data_input_bns_.size(), 0);
    CHECK_EQ(model_input_bns.size(), 1);
    model_input_bn_ = model_input_bns.at(0);
  }

  const std::string Description() const override {
    return op().op_name() + ": (B, S(0), ...) -> (S(0), ...)";
  }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelContext* parallel_ctx) const override {
    const auto& sbp_infer_hint = SbpInferHint4Ibn(model_input_bn_);
    if (!sbp_infer_hint.is_model_broadcast()) { return MakeOpParallelMatchSignatureMismatch(); }
    bool parallel_policy_matched = (parallel_ctx->policy() == kDataParallel);
    bool parallel_num_matched = (parallel_ctx->parallel_num() == sbp_infer_hint.parallel_num());
    if (!parallel_policy_matched || !parallel_num_matched) {
      OpParallelMatchResult ret;
      if (!parallel_policy_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
        err->set_configured(parallel_ctx->policy());
        err->set_expected(kDataParallel);
      }
      if (!parallel_num_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
        err->set_configured(parallel_ctx->parallel_num());
        err->set_expected(parallel_num_matched);
      }
      return ret;
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : data_input_bns_) { (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0); }
    (*bn2sbp)[model_input_bn_].mutable_broadcast_parallel();
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
    }
  }

 private:
  std::vector<std::string> data_input_bns_;
  std::string model_input_bn_;
};

class SoleIbnOpModelSplitOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoleIbnOpModelSplitOpParallelSignature);
  ~SoleIbnOpModelSplitOpParallelSignature() override = default;

  SoleIbnOpModelSplitOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (S,) -> (S, ...)"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelContext* parallel_ctx) const override {
    const SbpInferHint& sbp_infer_hint = SbpInferHint4Ibn(op().SoleIbn());
    if (!(sbp_infer_hint.is_model_split()
          || (sbp_infer_hint.is_data_split() && sbp_infer_hint.split_axis() > 0))) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    int64_t expected_parallel_num = sbp_infer_hint.parallel_num();
    bool parallel_policy_matched = (parallel_ctx->policy() == kModelParallel);
    bool parallel_num_matched = (parallel_ctx->parallel_num() == expected_parallel_num);
    if (!(parallel_policy_matched && parallel_num_matched)) {
      OpParallelMatchResult ret;
      if (!parallel_policy_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
        err->set_configured(parallel_ctx->policy());
        err->set_expected(kModelParallel);
      }
      if (!parallel_num_matched) {
        auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
        err->set_configured(parallel_ctx->parallel_num());
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
      const ParallelContext* parallel_ctx) const override {
    if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
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
    std::vector<std::string> model_input_bns;
    for (const auto& bn : op->input_bns()) {
      if (op->IsInputBlobAllowedModelSplit(bn)) {
        model_input_bns.push_back(bn);
      } else {
        data_input_bns_.push_back(bn);
      }
    }
    CHECK_GT(data_input_bns_.size(), 0);
    CHECK_EQ(model_input_bns.size(), 1);
    model_input_bn_ = model_input_bns.at(0);
  }

  const std::string Description() const override {
    return op().op_name() + ": (B, S, ...) -> (S, ...)";
  }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelContext* parallel_ctx) const override {
    const SbpInferHint& model_sbp_infer_hint = SbpInferHint4Ibn(model_input_bn_);
    if (!(model_sbp_infer_hint.is_model_split()
          && IsValidSplit(model_sbp_infer_hint.split_axis()))) {
      return MakeOpParallelMatchSignatureMismatch();
    }
    if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : data_input_bns_) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
    (*bn2sbp)[model_input_bn_] = SbpInferHint4Ibn(model_input_bn_).sbp_parallel();
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(
          op().OutputBlobModelSplitAxis(SbpInferHint4Ibn, bn));
    }
  }

 private:
  bool IsValidSplit(int32_t axis) const { return axis != -1 && IsExpectedAxis_(axis); }

  const std::function<bool(int32_t)> IsExpectedAxis_;
  std::vector<std::string> data_input_bns_;
  std::string model_input_bn_;
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
