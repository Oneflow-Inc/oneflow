#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

// S(0) -> (S | B)
class VariableOpSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VariableOpSbpSignatureRule);
  ~VariableOpSbpSignatureRule() override = default;

  VariableOpSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": S(0) -> (S | B)"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kDataParallel) {
      return MakeSbpSigMatchSuccess();
    } else {
      return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const SbpSignature& sbp_sig_hint, SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    if (op().op_conf().variable_conf().has_tick()) {
      CHECK(SbpInferHint4Ibn("tick").is_data_split());
      (*bn2sbp)["tick"].mutable_split_parallel()->set_axis(0);
    }
    const auto& conf_bn2sbp = sbp_sig_hint.bn_in_op2sbp_parallel();
    if (conf_bn2sbp.find("out") == conf_bn2sbp.end()) {
      (*bn2sbp)["out"].mutable_broadcast_parallel();
    } else {
      (*bn2sbp)["out"] = conf_bn2sbp.at("out");
    }
  }
};

}  // namespace

void VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  if (op_conf().variable_conf().has_tick()) { EnrollInputBn("tick", false); }
  bool has_diff =
      (Global<JobDesc>::Get()->IsTrain() && op_conf().trainable())
      || Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf();
  EnrollOutputBn("out", has_diff)->set_is_mutable(true);
  EnrollModelBn(op_conf().variable_conf().model_name());
}

const PbMessage& VariableOp::GetCustomizedConf() const { return op_conf().variable_conf(); }

int32_t VariableOp::OutputBlobModelSplitAxis(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const std::string& obn) const {
  return op_conf().variable_conf().model_split_axis();
}

void VariableOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  BlobDesc* model_blob_desc = GetBlobDesc4BnInOp(variable_conf.model_name());
  model_blob_desc->mut_shape() = Shape(variable_conf.shape());
  model_blob_desc->set_data_type(variable_conf.has_data_type()
                                     ? variable_conf.data_type()
                                     : Global<JobDesc>::Get()->DefaultDataType());
  if (parallel_ctx->policy() == kModelParallel) {
    int32_t model_split_axis = variable_conf.model_split_axis();
    CHECK_GE(model_split_axis, 0);
    CHECK_LT(model_split_axis, model_blob_desc->shape().NumAxes());
    int64_t split_dim_num = model_blob_desc->shape().At(model_split_axis);
    BalancedSplitter bs(split_dim_num, parallel_ctx->parallel_num());
    model_blob_desc->mut_shape().Set(model_split_axis, bs.At(parallel_ctx->parallel_id()).size());
  } else {
    CHECK_EQ(parallel_ctx->policy(), kDataParallel);
  }
  *GetBlobDesc4BnInOp("out") = *model_blob_desc;
}

void VariableOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new VariableOpSbpSignatureRule(this));
}

void VariableOp::InferIsModelBlob4OutputBlobs(
    std::function<bool*(const std::string&)> IsModelBlob4BnInOp) const {
  *IsModelBlob4BnInOp("out") = true;
}

void VariableOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* conf) const {
  conf->mutable_variable_conf()->set_is_fw_inplace(*is_fw_inplace_);
  conf->mutable_variable_conf()->set_is_bw_inplace(*is_bw_inplace_);
}

void VariableOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = false;
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);

}  // namespace oneflow
