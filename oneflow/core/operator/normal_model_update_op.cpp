#include "oneflow/core/operator/naive_model_update_op.h"
#include "oneflow/core/operator/rmsprop_model_update_op.h"
#include "oneflow/core/operator/momentum_model_update_op.h"
#include "oneflow/core/operator/lars_model_update_op.h"
#include "oneflow/core/operator/adam_model_update_op.h"

namespace oneflow {

namespace {

class ModelUpdtOpBroadcastSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdtOpBroadcastSignatureRule);
  ~ModelUpdtOpBroadcastSignatureRule() override = default;

  ModelUpdtOpBroadcastSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override {
    return op().op_name() + ": (B, ...) -> (B, ...)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& model_sbp_infer_hint = SbpInferHint4Ibn("model");
    CHECK(model_sbp_infer_hint.is_model_blob());
    if (model_sbp_infer_hint.sbp_parallel().has_broadcast_parallel()) {
      // TODO: CHECK(parallel_desc.EqualsIgnoringPolicy(model_sbp_infer_hint.parallel_desc()));
      return MakeSbpSigMatchSuccess();
    } else {
      return MakeSbpSigMatchSignatureMismatch();
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

class ModelUpdtOpSplitSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdtOpSplitSignatureRule);
  ~ModelUpdtOpSplitSignatureRule() override = default;

  ModelUpdtOpSplitSignatureRule(const Operator* op,
                                const HashSet<std::string>& always_broadcast_parallel_bns)
      : ParallelSbpSignatureRule(op),
        always_broadcast_parallel_bns_(always_broadcast_parallel_bns) {}

  const std::string Description() const override {
    return op().op_name() + ": (S(0), ...) -> (S(0), ...)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& model_sbp_infer_hint = SbpInferHint4Ibn("model");
    CHECK(model_sbp_infer_hint.is_model_blob());
    if (model_sbp_infer_hint.sbp_parallel().has_split_parallel()) {
      return MakeSbpSigMatchSuccess();
    } else {
      return MakeSbpSigMatchSignatureMismatch();
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    auto SetSbpParallel = [&](const std::string& bn) {
      if (always_broadcast_parallel_bns_.find(bn) == always_broadcast_parallel_bns_.end()) {
        (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
      } else {
        (*bn2sbp)[bn].mutable_broadcast_parallel();
      }
    };
    for (const auto& bn : op().input_bns()) { SetSbpParallel(bn); }
    for (const auto& bn : op().output_bns()) { SetSbpParallel(bn); }
  }

 private:
  HashSet<std::string> always_broadcast_parallel_bns_;
};

}  // namespace

std::unique_ptr<const SbpSignatureRule> MakeModelUpdtOpBroadcastSignatureRule(const Operator* op) {
  return std::make_unique<const ModelUpdtOpBroadcastSignatureRule>(op);
}

std::unique_ptr<const SbpSignatureRule> MakeModelUpdtOpSplitSignatureRule(
    const Operator* op, const HashSet<std::string>& always_broadcast_parallel_bns) {
  return std::make_unique<const ModelUpdtOpSplitSignatureRule>(op, always_broadcast_parallel_bns);
}

void NormalModelUpdtOp::InitFromOpConf() {
  EnrollInputBn("model_diff", false);
  EnrollInputBn("total_instance_num_diff", false);
  const JobDesc* g_job_conf = Global<JobDesc>::Get();
  if (g_job_conf->IsTrain()) {
    EnrollOutputBn("model", false);
  } else if (g_job_conf->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    EnrollInputBn("model", false)->set_is_mutable(true);
  } else {
    UNIMPLEMENTED();
  }
  const PbMessage& conf = this->GetCustomizedConf();
  const auto& user_conf = *GetMsgPtrFromPbMessage<NormalModelUpdateOpUserConf>(conf, "user_conf");
  if (user_conf.has_clip_conf() && user_conf.clip_conf().has_clip_by_global_norm()) {
    EnrollDataTmpBn("data_tmp");
  }
  MdUpdtVirtualInitFromOpConf();
}

void NormalModelUpdtOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const PbMessage& conf = this->GetCustomizedConf();
  const auto& user_conf = *GetMsgPtrFromPbMessage<NormalModelUpdateOpUserConf>(conf, "user_conf");
  if (user_conf.has_clip_conf() && user_conf.clip_conf().has_clip_by_global_norm()) {
    *GetBlobDesc4BnInOp("data_tmp") = *GetBlobDesc4BnInOp("model_diff");
    GetBlobDesc4BnInOp("data_tmp")->mut_shape() = Shape({1});
  }
  MdUpdtVirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

const PbMessage& NormalModelUpdtOp::GetCustomizedConf() const {
  return op_conf().normal_mdupdt_conf();
}

LogicalBlobId NormalModelUpdtOp::obn2lbi(const std::string& output_bn) const {
  const google::protobuf::Descriptor* desc = GetCustomizedConf().GetDescriptor();
  const google::protobuf::FieldDescriptor* fd = desc->FindFieldByName(output_bn);
  CHECK(fd);
  return GenLogicalBlobId(GetValFromCustomizedConf<std::string>(output_bn));
}

void NormalModelUpdtOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(MakeModelUpdtOpBroadcastSignatureRule(this));
  HashSet<std::string> broadcast_parallel_bns = AlwaysBroadcastParallelBns();
  broadcast_parallel_bns.emplace("total_instance_num_diff");
  rules->emplace_back(MakeModelUpdtOpSplitSignatureRule(this, broadcast_parallel_bns));
}

REGISTER_OP_CREATOR(OperatorConf::kNormalMdupdtConf, [](const OperatorConf& op_conf) -> Operator* {
  return NewObj<NormalModelUpdtOp>(op_conf.normal_mdupdt_conf().user_conf().normal_mdupdt_case());
});

}  // namespace oneflow
