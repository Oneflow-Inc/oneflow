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

std::unique_ptr<const OpParallelSignature> MakeDataSplitOpParallelSignature(const Operator* op) {
  std::string data_split_desc = op->op_name() + ": (S(0), ...) -> (S(0), ...)";
  auto IsMatched = [op](const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
                        const ParallelContext* parallel_ctx) {
    bool is_data_split = true;
    for (const auto& bn : op->input_bns()) {
      const LbpdHint& lbpd_hint = LbpdHint4BnInOp(bn);
      if (!lbpd_hint.is_data_blob()) {
        is_data_split = false;
        break;
      }
    }
    if (!is_data_split) { return MakeOpParallelMatchSignatureMismatch(); }
    if (parallel_ctx->policy() == kDataParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kDataParallel);
  };
  auto GenDataSplitSignature =
      [op](const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
           HashMap<std::string, LogicalBlobParallelDesc>* signature) {
        for (const auto& bn : op->input_bns()) {
          (*signature)[bn].mutable_split_parallel()->set_axis(0);
        }
        for (const auto& bn : op->output_bns()) {
          (*signature)[bn].mutable_split_parallel()->set_axis(0);
        }
      };
  return std::make_unique<OpParallelSignature>(data_split_desc, IsMatched, GenDataSplitSignature);
}

std::unique_ptr<const OpParallelSignature> MakeCloneOpParallelSignature(const Operator* op) {
  std::string clone_desc = op->op_name() + ": (C,) -> (C, ...)";
  auto IsSoleIbnCloned =
      [op](const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
           const ParallelContext* parallel_ctx) {
        if (!op->IsSoleInputBlobAllowedModelSplit()) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        if (!LbpdHint4BnInOp(op->SoleIbn()).has_model_clone()) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        int64_t expected_parallel_num = LbpdHint4BnInOp(op->SoleIbn()).parallel_num();
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
  auto GenCloneSignature = [op](const std::function<const LbpdHint&(const std::string&)>&,
                                HashMap<std::string, LogicalBlobParallelDesc>* signature) {
    for (const auto& bn : op->input_bns()) { (*signature)[bn].mutable_clone_parallel(); }
    for (const auto& bn : op->output_bns()) { (*signature)[bn].mutable_clone_parallel(); }
  };
  return std::make_unique<OpParallelSignature>(clone_desc, IsSoleIbnCloned, GenCloneSignature);
}

std::unique_ptr<const OpParallelSignature> MakeModelSplitOpParallelSignature(const Operator* op) {
  if (op->IsSoleInputBlobAllowedModelSplit()) {
    std::string desc = op->op_name() + ": (S,) -> (S, ...)";
    auto IsModelSplit =
        [op](const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
             const ParallelContext* parallel_ctx) {
          const LbpdHint& lbpd_hint = LbpdHint4BnInOp(op->SoleIbn());
          if (!(lbpd_hint.has_model_split()
                || (lbpd_hint.has_data_split() && lbpd_hint.data_split().axis() > 0))) {
            return MakeOpParallelMatchSignatureMismatch();
          }
          int64_t expected_parallel_num = lbpd_hint.parallel_num();
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
        [op](const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
             HashMap<std::string, LogicalBlobParallelDesc>* signature) {
          auto GetSplitParallel = [&](const std::string& bn) -> const SplitParallel& {
            if (LbpdHint4BnInOp(op->SoleIbn()).has_model_split()) {
              return LbpdHint4BnInOp(bn).model_split();
            } else if (LbpdHint4BnInOp(op->SoleIbn()).has_data_split()) {
              return LbpdHint4BnInOp(bn).data_split();
            } else {
              UNIMPLEMENTED();
            }
          };
          *((*signature)[op->SoleIbn()].mutable_split_parallel()) = GetSplitParallel(op->SoleIbn());
          for (const auto& bn : op->output_bns()) {
            *((*signature)[bn].mutable_split_parallel()) = GetSplitParallel(bn);
          }
        };
    return std::make_unique<OpParallelSignature>(desc, IsModelSplit, GenModelSplitSignature);
  } else {
    CHECK(!op->model_bns().empty() || !op->const_model_bns().empty());
    std::string desc = op->op_name() + ": (C, ...) -> (S, ...)";
    auto IsModelSplit = [op](const std::function<const LbpdHint&(const std::string&)>&,
                             const ParallelContext* parallel_ctx) {
      if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
      return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
    };
    auto GenModelSplitSignature =
        [op](const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
             HashMap<std::string, LogicalBlobParallelDesc>* signature) {
          for (const auto& bn : op->input_bns()) { (*signature)[bn].mutable_clone_parallel(); }
          for (const auto& bn : op->output_bns()) {
            *((*signature)[bn].mutable_split_parallel()) = LbpdHint4BnInOp(bn).data_split();
          }
        };
    return std::make_unique<OpParallelSignature>(desc, IsModelSplit, GenModelSplitSignature);
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
          const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
          const ParallelContext* parallel_ctx) {
        const auto& lbpd_hint = LbpdHint4BnInOp(model_input_bn);
        if (!lbpd_hint.has_model_clone()) { return MakeOpParallelMatchSignatureMismatch(); }
        bool parallel_policy_matched = (parallel_ctx->policy() == kDataParallel);
        bool parallel_num_matched = (parallel_ctx->parallel_num() == lbpd_hint.parallel_num());
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
                          const std::function<const LbpdHint&(const std::string&)>&,
                          HashMap<std::string, LogicalBlobParallelDesc>* signature) {
    for (const auto& bn : data_input_bns) {
      (*signature)[bn].mutable_split_parallel()->set_axis(0);
    }
    (*signature)[model_input_bn].mutable_clone_parallel();
    for (const auto& bn : op->output_bns()) {
      (*signature)[bn].mutable_split_parallel()->set_axis(0);
    }
  };
  return std::make_unique<OpParallelSignature>(desc, GetMatchResult, GenSignature);
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
          const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
          const ParallelContext* parallel_ctx) {
        const LbpdHint& model_lbpd_hint = LbpdHint4BnInOp(model_input_bn);
        if (!(model_lbpd_hint.has_model_split()
              && IsValidSplit(model_lbpd_hint.model_split().axis()))) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
        return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
      };
  auto GenSignature = [op, data_input_bns, model_input_bn](
                          const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
                          HashMap<std::string, LogicalBlobParallelDesc>* signature) {
    for (const auto& bn : data_input_bns) { (*signature)[bn].mutable_clone_parallel(); }
    *((*signature)[model_input_bn].mutable_split_parallel()) =
        LbpdHint4BnInOp(model_input_bn).model_split();
    for (const auto& bn : op->output_bns()) {
      *((*signature)[bn].mutable_split_parallel()) = LbpdHint4BnInOp(bn).data_split();
    }
  };
  return std::make_unique<OpParallelSignature>(desc, GetMatchResult, GenSignature);
}

}  // namespace oneflow
