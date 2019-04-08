#include "oneflow/core/operator/normalization_grad_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

class NormalizationGradDataParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationGradDataParallelSbpSignatureRule);
  ~NormalizationGradDataParallelSbpSignatureRule() override = default;

  explicit NormalizationGradDataParallelSbpSignatureRule(const Operator* op)
      : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (S, B) -> (S, P)"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kDataParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const NormalizationGradOpConf& conf = op().op_conf().normalization_grad_conf();
    (*bn2sbp)["dy"].mutable_split_parallel()->set_axis(0);
    if (conf.has_dx()) { (*bn2sbp)["dx"].mutable_split_parallel()->set_axis(0); }
    if (conf.has_gamma()) { (*bn2sbp)["gamma"].mutable_broadcast_parallel(); }
    if (conf.has_mean()) { (*bn2sbp)["mean"].mutable_broadcast_parallel(); }
    if (conf.has_inv_variance()) { (*bn2sbp)["inv_variance"].mutable_broadcast_parallel(); }
    if (conf.has_gamma_diff()) { (*bn2sbp)["gamma_diff"].mutable_partial_sum_parallel(); }
    if (conf.has_beta_diff()) { (*bn2sbp)["beta_diff"].mutable_partial_sum_parallel(); }
  }
};

}  // namespace

void NormalizationGradOp::InitFromOpConf() {
  const NormalizationGradOpConf& conf = op_conf().normalization_grad_conf();
#ifdef WITH_CUDA
  if (DevIsGpuAndEnableCudnn()) { CHECK_GE(conf.epsilon(), CUDNN_BN_MIN_EPSILON); }
#endif
  EnrollInputBn("dy", false);
  EnrollInputBn("x", false);
  CHECK_EQ(conf.has_mean(), conf.has_inv_variance());
  if (conf.has_mean() || conf.has_inv_variance()) {
    EnrollInputBn("mean", false);
    EnrollInputBn("inv_variance", false);
  }
  if (conf.has_gamma()) {
    EnrollInputBn("gamma", false);
  } else {
    if (DevIsGpuAndEnableCudnn()) {
      EnrollConstBufBn("gamma");
    } else {
      UNIMPLEMENTED();
    }
  }
  if (conf.has_dx()) {
    EnrollOutputBn("dx", false);
  } else {
    if (DevIsGpuAndEnableCudnn()) {
      EnrollOutputBn("dx", false);
    } else {
      UNIMPLEMENTED();
    }
  }
  if (conf.has_gamma_diff()) {
    EnrollOutputBn("gamma_diff", false);
  } else {
    if (DevIsGpuAndEnableCudnn()) {
      EnrollOutputBn("gamma_diff", false);
    } else {
      UNIMPLEMENTED();
    }
  }
  if (conf.has_beta_diff()) {
    EnrollOutputBn("beta_diff", false);
  } else {
    if (DevIsGpuAndEnableCudnn()) {
      EnrollOutputBn("beta_diff", false);
    } else {
      UNIMPLEMENTED();
    }
  }
}

const PbMessage& NormalizationGradOp::GetCustomizedConf() const {
  return op_conf().normalization_grad_conf();
}

void NormalizationGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NormalizationGradOpConf& conf = op_conf().normalization_grad_conf();
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  CHECK_EQ(dy->data_type(), x->data_type());
  CHECK_EQ(dy->shape(), x->shape());
  const DataType data_type = dy->data_type();
  BlobDesc* dx = GetBlobDesc4BnInOp("dx");
  if (dx) { *dx = *x; }
  const Shape param_shape({x->shape().At(conf.axis())});
  const auto CheckParamBlobDesc = [&](const std::string& bn) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);
    if (blob_desc != nullptr) {
      CHECK_EQ(blob_desc->data_type(), data_type);
      CHECK_EQ(blob_desc->shape(), param_shape);
    }
  };
  const auto SetParamBlobDesc = [&](const std::string& bn) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);
    if (blob_desc != nullptr) {
      blob_desc->set_data_type(data_type);
      blob_desc->mut_shape() = param_shape;
    }
  };
  CheckParamBlobDesc("mean");
  CheckParamBlobDesc("inv_variance");
  CheckParamBlobDesc("gamma");
  SetParamBlobDesc("gamma_diff");
  SetParamBlobDesc("beta_diff");
}

void NormalizationGradOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new NormalizationGradDataParallelSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kNormalizationGradConf, NormalizationGradOp);

}  // namespace oneflow
