#include "oneflow/core/operator/conv_filter_grad_op.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"

namespace oneflow {

namespace {

class ConvFilterGradDataParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradDataParallelSbpSignatureRule);
  ~ConvFilterGradDataParallelSbpSignatureRule() override = default;

  explicit ConvFilterGradDataParallelSbpSignatureRule(const Operator* op)
      : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (MB, DS) -> P"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kDataParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["dy"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["x"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["filter_diff"].mutable_partial_sum_parallel();
  }
};

class ConvFilterGradModelParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradModelParallelSbpSignatureRule);
  ~ConvFilterGradModelParallelSbpSignatureRule() override = default;

  explicit ConvFilterGradModelParallelSbpSignatureRule(const Operator* op)
      : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (MS, DB) -> S"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kModelParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const ConvConf& conf = op().op_conf().conv_filter_grad_conf().conv_conf();
    if (conf.data_format() == "channels_first") {
      (*bn2sbp)["dy"].mutable_split_parallel()->set_axis(1);
    } else if (conf.data_format() == "channels_last") {
      (*bn2sbp)["dy"].mutable_split_parallel()->set_axis(1 + conf.num_spatial_dims());
    } else {
      UNIMPLEMENTED();
    }
    (*bn2sbp)["filter_diff"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["x"].mutable_broadcast_parallel();
  }
};

}  // namespace

const PbMessage& ConvFilterGradOp::GetCustomizedConf() const {
  return op_conf().conv_filter_grad_conf();
}

void ConvFilterGradOp::InitFromOpConf() {
  CHECK(op_conf().has_conv_filter_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("x", false);
  EnrollOutputBn("filter_diff", false);
  if (DevIsGpuAndEnableCudnn()) {
    EnrollFwBufBn("buf");
  } else {
    UNIMPLEMENTED();
  }
}

void ConvFilterGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, int64_t record_piece_size,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const ConvFilterGradOpConf& conf = this->op_conf().conv_filter_grad_conf();
  const ConvConf& conv_conf = conf.conv_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  BlobDesc* filter_diff = GetBlobDesc4BnInOp("filter_diff");
  const int32_t num_spatial_dims = conf.conv_conf().num_spatial_dims();
  CHECK_GE(num_spatial_dims, 1);
  CHECK_LE(num_spatial_dims, 3);
  CHECK_EQ(dy->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ(x->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ(x->data_type(), dy->data_type());
  std::vector<int64_t> filter_diff_dim_vec;
  if (conv_conf.data_format() == "channels_first") {
    filter_diff_dim_vec.push_back(dy->shape().At(1));
    filter_diff_dim_vec.push_back(x->shape().At(1));
    filter_diff_dim_vec.insert(filter_diff_dim_vec.end(), conv_conf.kernel_size().cbegin(),
                               conv_conf.kernel_size().cend());
  } else if (conv_conf.data_format() == "channels_last") {
    filter_diff_dim_vec.push_back(dy->shape().dim_vec().back());
    filter_diff_dim_vec.insert(filter_diff_dim_vec.end(), conv_conf.kernel_size().cbegin(),
                               conv_conf.kernel_size().cend());
    filter_diff_dim_vec.push_back(x->shape().dim_vec().back());
  } else {
    UNIMPLEMENTED();
  }
  filter_diff->mut_shape() = Shape(filter_diff_dim_vec);
  filter_diff->set_data_type(x->data_type());

  if (DevIsGpuAndEnableCudnn()) {
#ifdef WITH_CUDA
    ConvOpCtx* conv_op_ctx = new ConvOpCtx();
    EnrollOpCtx(conv_op_ctx);
    CHECK(Global<CudnnConvCtxCache>::Get()->FindCudnnConvAlgoCtxWithConfig(
        *x, *dy, *filter_diff, conv_conf, cudnn_buf_limit_byte(),
        &conv_op_ctx->cudnn_conv_algo_ctx));
    CHECK(conv_op_ctx->cudnn_conv_algo_ctx.bwd_filter_algo_found);
    BlobDesc* cudnn_buf = GetBlobDesc4BnInOp("buf");
    cudnn_buf->set_data_type(DataType::kChar);
    cudnn_buf->mut_shape() =
        Shape({static_cast<int64_t>(conv_op_ctx->cudnn_conv_algo_ctx.bwd_filter_ws_size)});
#else
    UNIMPLEMENTED();
#endif
  } else {
    UNIMPLEMENTED();
  }
}

int32_t ConvFilterGradOp::OutputBlobModelSplitAxis(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const std::string& obn) const {
  return 0;
}

void ConvFilterGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  if (DevIsGpuAndEnableCudnn()) {
#ifdef WITH_CUDA
    const ConvOpCtx* conv_op_ctx = dynamic_cast<const ConvOpCtx*>(op_ctx);
    kernel_conf->mutable_conv_filter_grad_conf()->set_cudnn_bwd_filter_algo(
        conv_op_ctx->cudnn_conv_algo_ctx.bwd_filter_algo);
#else
    UNIMPLEMENTED();
#endif  // WITH_CUDA
  } else {
    UNIMPLEMENTED();
  }
}

void ConvFilterGradOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new ConvFilterGradDataParallelSbpSignatureRule(this));
  rules->emplace_back(new ConvFilterGradModelParallelSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kConvFilterGradConf, ConvFilterGradOp);

}  // namespace oneflow
