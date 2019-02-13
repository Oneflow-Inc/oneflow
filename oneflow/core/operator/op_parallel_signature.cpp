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

  DataSplitOpParallelSignature(const Operator* op) : OpParallelSignature(), op_(op) {}

  const std::string Description() const override {
    return op_->op_name() + ": (S(0), ...) -> (S(0), ...)";
  }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    bool is_data_split = true;
    for (const auto& bn : op_->input_bns()) {
      const SbpInferHint& sbp_infer_hint = SbpInferHint4BnInOp(bn);
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
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : op_->input_bns()) { (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0); }
    for (const auto& bn : op_->output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
    }
  }

 private:
  const Operator* op_;
};

}  // namespace

std::unique_ptr<const OpParallelSignature> MakeDataSplitOpParallelSignature(const Operator* op) {
  return std::unique_ptr<const OpParallelSignature>(new DataSplitOpParallelSignature(op));
}

std::unique_ptr<const OpParallelSignature> MakeCloneOpParallelSignature(const Operator* op) {
  std::string clone_desc = op->op_name() + ": (C,) -> (C, ...)";
  auto IsSoleIbnCloned =
      [op](const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
           const ParallelContext* parallel_ctx) {
        if (!op->IsSoleInputBlobAllowedModelSplit()) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        if (!SbpInferHint4BnInOp(op->SoleIbn()).is_model_broadcast()) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        int64_t expected_parallel_num = SbpInferHint4BnInOp(op->SoleIbn()).parallel_num();
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
      };
  auto GenCloneSignature = [op](const std::function<const SbpInferHint&(const std::string&)>&,
                                HashMap<std::string, SbpParallel>* signature) {
    for (const auto& bn : op->input_bns()) { (*signature)[bn].mutable_broadcast_parallel(); }
    for (const auto& bn : op->output_bns()) { (*signature)[bn].mutable_broadcast_parallel(); }
  };
  return std::unique_ptr<const OpParallelSignature>(
      new LambdaOpParallelSignature(clone_desc, IsSoleIbnCloned, GenCloneSignature));
}

std::unique_ptr<const OpParallelSignature> MakeModelSplitOpParallelSignature(const Operator* op) {
  if (op->IsSoleInputBlobAllowedModelSplit()) {
    std::string desc = op->op_name() + ": (S,) -> (S, ...)";
    auto IsModelSplit =
        [op](const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
             const ParallelContext* parallel_ctx) {
          const SbpInferHint& sbp_infer_hint = SbpInferHint4BnInOp(op->SoleIbn());
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
        };
    auto GenModelSplitSignature =
        [op](const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
             HashMap<std::string, SbpParallel>* signature) {
          (*signature)[op->SoleIbn()].mutable_split_parallel()->set_axis(
              SbpInferHint4BnInOp(op->SoleIbn()).split_axis());
          for (const auto& bn : op->output_bns()) {
            (*signature)[bn].mutable_split_parallel()->set_axis(
                SbpInferHint4BnInOp(bn).split_axis());
          }
        };
    return std::unique_ptr<const OpParallelSignature>(
        new LambdaOpParallelSignature(desc, IsModelSplit, GenModelSplitSignature));
  } else {
    CHECK(!op->model_bns().empty() || !op->const_model_bns().empty());
    std::string desc = op->op_name() + ": (C, ...) -> (S, ...)";
    auto IsModelSplit = [op](const std::function<const SbpInferHint&(const std::string&)>&,
                             const ParallelContext* parallel_ctx) {
      if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
      return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
    };
    auto GenModelSplitSignature =
        [op](const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
             HashMap<std::string, SbpParallel>* signature) {
          for (const auto& bn : op->input_bns()) { (*signature)[bn].mutable_broadcast_parallel(); }
          for (const auto& bn : op->output_bns()) {
            (*signature)[bn].mutable_split_parallel()->set_axis(
                SbpInferHint4BnInOp(bn).split_axis());
          }
        };
    return std::unique_ptr<const OpParallelSignature>(
        new LambdaOpParallelSignature(desc, IsModelSplit, GenModelSplitSignature));
  }
}

std::unique_ptr<const OpParallelSignature> MakeOpParallelSignature_DS_MC_2_DS(const Operator* op) {
  std::string desc = op->op_name() + ": (C, S(0), ...) -> (S(0), ...)";
  std::vector<std::string> data_input_bns;
  std::vector<std::string> model_input_bns;
  for (const auto& bn : op->input_bns()) {
    if (op->IsInputBlobAllowedModelSplit(bn)) {
      model_input_bns.push_back(bn);
    } else {
      data_input_bns.push_back(bn);
    }
  }
  CHECK_GT(data_input_bns.size(), 0);
  CHECK_EQ(model_input_bns.size(), 1);
  std::string model_input_bn = model_input_bns.at(0);
  auto GetMatchResult =
      [op, data_input_bns, model_input_bn](
          const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
          const ParallelContext* parallel_ctx) {
        const auto& sbp_infer_hint = SbpInferHint4BnInOp(model_input_bn);
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
      };
  auto GenSignature = [op, data_input_bns, model_input_bn](
                          const std::function<const SbpInferHint&(const std::string&)>&,
                          HashMap<std::string, SbpParallel>* signature) {
    for (const auto& bn : data_input_bns) {
      (*signature)[bn].mutable_split_parallel()->set_axis(0);
    }
    (*signature)[model_input_bn].mutable_broadcast_parallel();
    for (const auto& bn : op->output_bns()) {
      (*signature)[bn].mutable_split_parallel()->set_axis(0);
    }
  };
  return std::unique_ptr<const OpParallelSignature>(
      new LambdaOpParallelSignature(desc, GetMatchResult, GenSignature));
}

std::unique_ptr<const OpParallelSignature> MakeOpParallelSignature_DC_MS_2_MS(
    const Operator* op, std::function<bool(int32_t)> IsExpectedAxis) {
  auto IsValidSplit = [IsExpectedAxis](int32_t x) { return x != -1 && IsExpectedAxis(x); };
  std::string desc = op->op_name() + ": (C, S, ...) -> (S, ...)";
  std::vector<std::string> data_input_bns;
  std::vector<std::string> model_input_bns;
  for (const auto& bn : op->input_bns()) {
    if (op->IsInputBlobAllowedModelSplit(bn)) {
      model_input_bns.push_back(bn);
    } else {
      data_input_bns.push_back(bn);
    }
  }
  CHECK_GT(data_input_bns.size(), 0);
  CHECK_EQ(model_input_bns.size(), 1);
  std::string model_input_bn = model_input_bns.at(0);
  auto GetMatchResult =
      [op, data_input_bns, model_input_bn, IsValidSplit](
          const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
          const ParallelContext* parallel_ctx) {
        const SbpInferHint& model_sbp_infer_hint = SbpInferHint4BnInOp(model_input_bn);
        if (!(model_sbp_infer_hint.is_model_split()
              && IsValidSplit(model_sbp_infer_hint.split_axis()))) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
        return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
      };
  auto GenSignature =
      [op, data_input_bns, model_input_bn](
          const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
          HashMap<std::string, SbpParallel>* signature) {
        for (const auto& bn : data_input_bns) { (*signature)[bn].mutable_broadcast_parallel(); }
        (*signature)[model_input_bn].mutable_split_parallel()->set_axis(
            SbpInferHint4BnInOp(model_input_bn).split_axis());
        for (const auto& bn : op->output_bns()) {
          (*signature)[bn].mutable_split_parallel()->set_axis(SbpInferHint4BnInOp(bn).split_axis());
        }
      };
  return std::unique_ptr<const OpParallelSignature>(
      new LambdaOpParallelSignature(desc, GetMatchResult, GenSignature));
}

}  // namespace oneflow
