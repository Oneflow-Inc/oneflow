#include "oneflow/core/operator/conv_data_grad_op.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_conv_ctx_cache.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

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
    EnrollTmpBn("buf");
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
    size_t buf_size = std::max(size_t(1), conv_op_ctx->cudnn_conv_algo_ctx.bwd_data_ws_size);
    cudnn_buf->mut_shape() = Shape({static_cast<int64_t>(buf_size)});
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

void ConvDataGradOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  CHECK(*HasBatchDim4BnInOp("dy"));
  CHECK(*HasBatchDim4BnInOp("x_like"));
  CHECK(*HasBatchDim4BnInOp("filter") == false);
  *HasBatchDim4BnInOp("dx") = true;
}

void ConvDataGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("dy", 0)
      .Broadcast("filter")
      .Split("x_like", 0)
      .Split("dx", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kConvDataGradConf, ConvDataGradOp);

}  // namespace oneflow
