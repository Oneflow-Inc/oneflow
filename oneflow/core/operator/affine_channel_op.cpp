#include "oneflow/core/operator/affine_channel_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void AffineChannelOp::InitFromOpConf() {
  const auto& conf = op_conf().affine_channel_conf();
  CHECK_GE(conf.momentum(), 0);
  CHECK_LE(conf.momentum(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollConstBufBn("zero_buf");
  if (conf.center()) {
    EnrollModelBn("beta");
  } else {
    EnrollDataTmpBn("beta_diff");
  }
  if (conf.scale()) {
    EnrollModelBn("gamma");
  } else {
    EnrollDataTmpBn("gamma_diff");
  }
}

const PbMessage& AffineChannelOp::GetCustomizedConf() const {
  return op_conf().affine_channel_conf();
}

void AffineChannelOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, std::function<void(OpContext*)> EnrollOpCtx) const {
  const auto& conf = op_conf().affine_channel_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const DataType in_data_type = in_blob_desc->data_type();
  CHECK_EQ(in_data_type, Global<JobDesc>::Get()->DefaultDataType());
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  int32_t in_dims = in_blob_desc->shape().NumAxes();
#ifdef WITH_CUDA
  if (device_type() == DeviceType::kGPU && CUDNN_VERSION >= 5000 && in_data_type == DataType::kFloat
      && in_dims >= 4 && in_dims <= 5 && (conf.axis() == 1 || conf.axis() == in_dims - 1)) {
    InferBlobDescsForCudnn(GetBlobDesc4BnInOp);
    return;
  }
#endif
  AffineChannelOpCtx* op_ctx = NewAffineChannelOpCtx(in_blob_desc->shape());
  EnrollOpCtx(op_ctx);
  InferParamBlobDescs(GetBlobDesc4BnInOp, conf, in_blob_desc->shape().At(conf.axis()), in_data_type,
                      false);
}

void AffineChannelOp::InferParamBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const AffineChannelOpConf& conf, int64_t norm_part_num, DataType in_data_type,
    bool use_cudnn) const {
  BlobDesc blob_desc(Shape({norm_part_num}), in_data_type, false, false, 1);
  std::list<std::string> blob_names = {"zero_buf"};
  if (conf.center()) {
    blob_names.push_back("beta");
  } else {
    blob_names.push_back("beta_diff");
  }
  if (conf.scale()) {
    blob_names.push_back("gamma");
  } else {
    blob_names.push_back("gamma_diff");
  }
  for (const auto& bn_in_op : blob_names) { *GetBlobDesc4BnInOp(bn_in_op) = blob_desc; }
}

void AffineChannelOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  AffineChannelKernelConf* conf = kernel_conf->mutable_affine_channel_conf();
  const auto* ctx = dynamic_cast<const AffineChannelOpCtx*>(op_ctx);
#ifdef WITH_CUDA
  if (ctx == nullptr) {
    VirtualGenKernelConfForCudnn(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf);
    return;
  }
#endif
  conf->set_axis(ctx->axis);
  conf->set_use_cudnn(false);
}

#ifdef WITH_CUDA
void AffineChannelOp::InferBlobDescsForCudnn(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const {
  const auto& conf = op_conf().affine_channel_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const DataType in_data_type = in_blob_desc->data_type();
  CHECK(conf.scale() && conf.center()) << "Cudnn batch norm must use scale and center";
  InferParamBlobDescs(GetBlobDesc4BnInOp, conf, in_blob_desc->shape().At(conf.axis()), in_data_type,
                      true);
}

void AffineChannelOp::VirtualGenKernelConfForCudnn(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  AffineChannelKernelConf* conf = kernel_conf->mutable_affine_channel_conf();
  conf->set_use_cudnn(true);
  GetBlobDesc4BnInOp("in")->shape().ToProto(conf->mutable_in());
#if (CUDNN_VERSION >= 7000)
  conf->set_cudnn_bn_mode(CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
#else
  conf->set_cudnn_bn_mode(CUDNN_BATCHNORM_SPATIAL);
#endif
}
#endif

void AffineChannelOp::VirtualFixParallelDesc(ParallelDesc* pr_desc) const {
  pr_desc->set_policy(ParallelPolicy::kDataParallel);
}

AffineChannelOpCtx* AffineChannelOp::NewAffineChannelOpCtx(const Shape& in_shape) const {
  AffineChannelOpCtx* op_ctx = new AffineChannelOpCtx();
  op_ctx->axis = op_conf().normalization_conf().axis();
  op_ctx->dims = in_shape.NumAxes();
  if (op_ctx->axis < 0) { op_ctx->axis += op_ctx->dims; }
  CHECK_GE(op_ctx->axis, 0);
  CHECK_LT(op_ctx->axis, op_ctx->dims);
  return op_ctx;
}

REGISTER_OP(OperatorConf::kAffineChannelConf, AffineChannelOp);

}  // namespace oneflow
