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

const OpParallelSignature MakeOpDataSplitParallelSignature(const Operator* op) {
  std::string data_split_desc = op->op_name() + ": (S(0), ...) -> (S(0), ...)";
  auto IsMatched =
      [op](const std::function<const BlobParallelType&(const std::string&)>& ProducerLbpd4Ibn,
           const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
           const ParallelContext* parallel_ctx) {
        OpParallelMatchResult default_ret;
        if (parallel_ctx->policy() == kDataParallel) {
          default_ret = MakeOpParallelMatchSuccess();
        } else {
          default_ret =
              MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kDataParallel);
        }
        if (op->IsSoleInputBlobAllowedModelSplit()) {
          const auto& producer_lbpd = ProducerLbpd4Ibn(op->SoleIbn());
          if (producer_lbpd.has_partial_sum_parallel()) { return default_ret; }
          if (producer_lbpd.has_split_parallel() && ModelSplitAxis4BnInOp(op->SoleIbn()) == -1) {
            return default_ret;
          }
          return MakeOpParallelMatchSignatureMismatch();
        } else {
          CHECK(op->model_bns().empty());
          CHECK(op->const_model_bns().empty());
          CHECK(op->forward_model_bns().empty());
          for (const std::string ibn : op->input_bns()) {
            if (op->IsInputBlobAllowedModelSplit(ibn)) { UNIMPLEMENTED(); }
          }
          return default_ret;
        }
      };
  auto GenDataSplitSignature =
      [op](const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
           HashMap<std::string, BlobParallelType>* signature) {
        for (const auto& bn : op->input_bns()) {
          (*signature)[bn].mutable_split_parallel()->set_axis(0);
        }
        for (const auto& bn : op->output_bns()) {
          (*signature)[bn].mutable_split_parallel()->set_axis(0);
        }
      };
  return OpParallelSignature(data_split_desc, IsMatched, GenDataSplitSignature);
}

const OpParallelSignature MakeOpCloneParallelSignature(const Operator* op) {
  std::string clone_desc = op->op_name() + ": (C,) -> (C, ...)";
  auto IsSoleIbnCloned =
      [op](const std::function<const BlobParallelType&(const std::string&)>& ProducerLbpd4Ibn,
           const std::function<int32_t(const std::string&)>&, const ParallelContext* parallel_ctx) {
        if (!op->IsSoleInputBlobAllowedModelSplit()) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        if (!ProducerLbpd4Ibn(op->SoleIbn()).has_clone_parallel()) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        int64_t expected_parallel_num = ProducerLbpd4Ibn(op->SoleIbn()).parallel_num();
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
  auto GenCloneSignature = [op](const std::function<int32_t(const std::string&)>&,
                                HashMap<std::string, BlobParallelType>* signature) {
    for (const auto& bn : op->input_bns()) { (*signature)[bn].mutable_clone_parallel(); }
    for (const auto& bn : op->output_bns()) { (*signature)[bn].mutable_clone_parallel(); }
  };
  return OpParallelSignature(clone_desc, IsSoleIbnCloned, GenCloneSignature);
}

const OpParallelSignature MakeOpModelSplitParallelSignature(const Operator* op) {
  if (op->IsSoleInputBlobAllowedModelSplit()) {
    std::string desc = op->op_name() + ": (S,) -> (S, ...)";
    auto IsModelSplit =
        [op](const std::function<const BlobParallelType&(const std::string&)>& ProducerLbpd4Ibn,
             const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
             const ParallelContext* parallel_ctx) {
          const auto& producer_lbpd = ProducerLbpd4Ibn(op->SoleIbn());
          if (!producer_lbpd.has_split_parallel()) {
            return MakeOpParallelMatchSignatureMismatch();
          }
          if (ModelSplitAxis4BnInOp(op->SoleIbn()) == -1) {
            return MakeOpParallelMatchSignatureMismatch();
          }
          int64_t expected_parallel_num = ProducerLbpd4Ibn(op->SoleIbn()).parallel_num();
          bool parallel_policy_matched = (parallel_ctx->policy() == kModelParallel);
          bool parallel_num_matched = (parallel_ctx->parallel_num() == expected_parallel_num);
          if (parallel_policy_matched && parallel_num_matched) {
            return MakeOpParallelMatchSuccess();
          } else {
            OpParallelMatchResult ret;
            if (!parallel_policy_matched) {
              auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_policy_error();
              err->set_configured(parallel_ctx->policy());
              err->set_expected(kModelParallel);
            } else {
              auto* err = ret.mutable_fail()->mutable_conf_error()->mutable_parallel_num_error();
              err->set_configured(parallel_ctx->parallel_num());
              err->set_expected(parallel_num_matched);
            }
            return ret;
          }
        };
    auto GenModelSplitSignature =
        [op](const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
             HashMap<std::string, BlobParallelType>* signature) {
          for (const auto& bn : op->input_bns()) {
            (*signature)[bn].mutable_split_parallel()->set_axis(ModelSplitAxis4BnInOp(bn));
          }
          for (const auto& bn : op->output_bns()) {
            (*signature)[bn].mutable_split_parallel()->set_axis(ModelSplitAxis4BnInOp(bn));
          }
        };
    return OpParallelSignature(desc, IsModelSplit, GenModelSplitSignature);
  } else {
    CHECK(!op->model_bns().empty() || !op->const_model_bns().empty());
    std::string desc = op->op_name() + ": (C, ...) -> (S, ...)";
    auto IsModelSplit = [op](const std::function<const BlobParallelType&(const std::string&)>&,
                             const std::function<int32_t(const std::string&)>&,
                             const ParallelContext* parallel_ctx) {
      bool parallel_policy_matched = (parallel_ctx->policy() == kModelParallel);
      if (parallel_policy_matched) {
        return MakeOpParallelMatchSuccess();
      } else {
        return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
      }
    };
    auto GenModelSplitSignature =
        [op](const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
             HashMap<std::string, BlobParallelType>* signature) {
          for (const auto& bn : op->input_bns()) { (*signature)[bn].mutable_clone_parallel(); }
          for (const auto& bn : op->output_bns()) {
            (*signature)[bn].mutable_split_parallel()->set_axis(ModelSplitAxis4BnInOp(bn));
          }
        };
    return OpParallelSignature(desc, IsModelSplit, GenModelSplitSignature);
  }
}

}  // namespace oneflow
