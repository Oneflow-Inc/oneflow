#include "oneflow/core/operator/conv_bias_grad_op.h"

namespace oneflow {

namespace {

class ConvBiasGradDataParallelSbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvBiasGradDataParallelSbpSignature);
  ~ConvBiasGradDataParallelSbpSignature() override = default;

  explicit ConvBiasGradDataParallelSbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(0) -> P"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kDataParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["dy"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["bias_diff"].mutable_partial_sum_parallel();
  }
};

class ConvBiasGradModelParallelSbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvBiasGradModelParallelSbpSignature);
  ~ConvBiasGradModelParallelSbpSignature() override = default;

  explicit ConvBiasGradModelParallelSbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S(chan) -> S"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kModelParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    const ConvBiasGradOpConf& conf = op().op_conf().conv_bias_grad_conf();
    if (conf.data_format() == "channels_first") {
      (*bn2sbp)["dy"].mutable_split_parallel()->set_axis(1);
    } else if (conf.data_format() == "channels_last") {
      (*bn2sbp)["dy"].mutable_split_parallel()->set_axis(1 + conf.num_spatial_dims());
    } else {
      UNIMPLEMENTED();
    }
    (*bn2sbp)["bias_diff"].mutable_split_parallel()->set_axis(0);
  }
};

}  // namespace

const PbMessage& ConvBiasGradOp::GetCustomizedConf() const {
  return op_conf().conv_bias_grad_conf();
}

void ConvBiasGradOp::InitFromOpConf() {
  CHECK(op_conf().has_conv_bias_grad_conf());
  EnrollInputBn("dy", false);
  EnrollOutputBn("bias_diff", false);
}

void ConvBiasGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const ConvBiasGradOpConf& conf = this->op_conf().conv_bias_grad_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  BlobDesc* bias_diff = GetBlobDesc4BnInOp("bias_diff");
  CHECK_GE(conf.num_spatial_dims(), 1);
  CHECK_LE(conf.num_spatial_dims(), 3);
  CHECK_EQ(dy->shape().NumAxes(), conf.num_spatial_dims() + 2);
  bias_diff->set_data_type(dy->data_type());
  if (conf.data_format() == "channels_first") {
    bias_diff->mut_shape() = Shape({dy->shape().At(1)});
  } else if (conf.data_format() == "channels_last") {
    bias_diff->mut_shape() = Shape({dy->shape().At(dy->shape().NumAxes() - 1)});
  } else {
    UNIMPLEMENTED();
  }
}

int32_t ConvBiasGradOp::OutputBlobModelSplitAxis(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const std::string& obn) const {
  return 0;
}

void ConvBiasGradOp::GetSbpSignatures(
    std::vector<std::unique_ptr<const SbpSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new ConvBiasGradDataParallelSbpSignature(this));
  op_parallel_signatures->emplace_back(new ConvBiasGradModelParallelSbpSignature(this));
}

REGISTER_OP(OperatorConf::kConvBiasGradConf, ConvBiasGradOp);

}  // namespace oneflow
