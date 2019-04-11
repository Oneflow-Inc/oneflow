#include "oneflow/core/operator/conv_data_grad_op.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"

namespace oneflow {

namespace {

class ConvDataGradDataParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradDataParallelSbpSignatureRule);
  ~ConvDataGradDataParallelSbpSignatureRule() override = default;

  explicit ConvDataGradDataParallelSbpSignatureRule(const Operator* op)
      : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (MB, DS) -> DS"; }

  const SbpSigMatchResult GetMatchResult(
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
    (*bn2sbp)["x_like"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["filter"].mutable_broadcast_parallel();
    (*bn2sbp)["dx"].mutable_split_parallel()->set_axis(0);
  }
};

class ConvDataGradModelParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradModelParallelSbpSignatureRule);
  ~ConvDataGradModelParallelSbpSignatureRule() override = default;

  explicit ConvDataGradModelParallelSbpSignatureRule(const Operator* op)
      : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (DS, MS) -> P"; }

  const SbpSigMatchResult GetMatchResult(
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
    (*bn2sbp)["filter"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["x_like"].mutable_broadcast_parallel();
    (*bn2sbp)["dx"].mutable_partial_sum_parallel();
  }
};

}  // namespace

const PbMessage& ConvDataGradOp::GetCustomizedConf() const {
  return op_conf().conv_data_grad_conf();
}

void ConvDataGradOp::InitFromOpConf() {
  CHECK(op_conf().has_conv_data_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("filter", false);
  EnrollInputBn("x_like", false)->set_use_header_only(true);
  EnrollOutputBn("dx", false);
  if (DevIsGpuAndEnableCudnn()) {
    EnrollFwBufBn("buf");
  } else {
    UNIMPLEMENTED();
  }
}

void ConvDataGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx, int64_t record_piece_size,
                                    std::function<void(OpContext*)> EnrollOpCtx) const {
  const ConvDataGradOpConf& conf = this->op_conf().conv_data_grad_conf();
  const ConvConf& conv_conf = conf.conv_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  const BlobDesc* filter = GetBlobDesc4BnInOp("filter");
  const BlobDesc* x_like = GetBlobDesc4BnInOp("x_like");
  const int32_t num_spatial_dims = conf.conv_conf().num_spatial_dims();
  CHECK_GE(num_spatial_dims, 1);
  CHECK_LE(num_spatial_dims, 3);
  CHECK_EQ(dy->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ(x_like->shape().NumAxes(), num_spatial_dims + 2);
  CHECK_EQ(x_like->data_type(), dy->data_type());
  BlobDesc* dx = GetBlobDesc4BnInOp("dx");
  dx->CopyMetaFrom(*x_like);
  if (DevIsGpuAndEnableCudnn()) {
#ifdef WITH_CUDA
    ConvOpCtx* conv_op_ctx = new ConvOpCtx();
    EnrollOpCtx(conv_op_ctx);
    CHECK(Global<CudnnConvCtxCache>::Get()->FindCudnnConvAlgoCtxWithConfig(
        *x_like, *dy, *filter, conv_conf, cudnn_buf_limit_byte(),
        &conv_op_ctx->cudnn_conv_algo_ctx));
    CHECK(conv_op_ctx->cudnn_conv_algo_ctx.bwd_data_algo_found);
    BlobDesc* cudnn_buf = GetBlobDesc4BnInOp("buf");
    cudnn_buf->set_data_type(DataType::kChar);
    cudnn_buf->mut_shape() =
        Shape({static_cast<int64_t>(conv_op_ctx->cudnn_conv_algo_ctx.bwd_data_ws_size)});
#else
    UNIMPLEMENTED();
#endif
  } else {
    UNIMPLEMENTED();
  }
}

void ConvDataGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  if (DevIsGpuAndEnableCudnn()) {
#ifdef WITH_CUDA
    const ConvOpCtx* conv_op_ctx = dynamic_cast<const ConvOpCtx*>(op_ctx);
    kernel_conf->mutable_conv_data_grad_conf()->set_cudnn_bwd_data_algo(
        conv_op_ctx->cudnn_conv_algo_ctx.bwd_data_algo);
#else
    UNIMPLEMENTED();
#endif  // WITH_CUDA
  } else {
    UNIMPLEMENTED();
  }
}

void ConvDataGradOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new ConvDataGradDataParallelSbpSignatureRule(this));
  rules->emplace_back(new ConvDataGradModelParallelSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kConvDataGradConf, ConvDataGradOp);

}  // namespace oneflow
