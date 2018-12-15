#include "oneflow/core/operator/layer_norm_op.h"

namespace oneflow {

namespace {

int64_t ShiftNegativeAxisIfNeed(const Shape& shape, int64_t axis) {
  const int64_t shifted = axis < 0 ? axis + shape.NumAxes() : axis;
  CHECK_GE(shifted, 0);
  CHECK_LT(shifted, shape.NumAxes());
  return shifted;
}

}  // namespace

void LayerNormOp::InitFromOpConf() {
  CHECK(op_conf().has_layer_norm_conf());
  const LayerNormOpConf& conf = op_conf().layer_norm_conf();
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (conf.center()) { EnrollModelBn("beta"); }
  if (conf.scale()) {
    EnrollModelBn("gamma");
    if (Global<JobDesc>::Get()->IsTrain()) { EnrollDataTmpBn("normalize_out"); }
  }
  EnrollDataTmpBn("cudnn_bn_mean");
  EnrollDataTmpBn("cudnn_bn_inv_variance");
  EnrollConstBufBn("cudnn_bn_scale_ones");
  EnrollConstBufBn("cudnn_bn_bias_zeros");
  EnrollBwBufBn("cudnn_bn_scale_diff_buf");
  EnrollBwBufBn("cudnn_bn_bias_diff_buf");
  if (conf.center() || conf.scale()) { EnrollBwBufBn("reusable_bw_buf"); }
}

void LayerNormOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  CHECK(parallel_ctx->policy() != kModelParallel);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in;
  const LayerNormOpConf& conf = op_conf().layer_norm_conf();
  const int64_t begin_params_axis = ShiftNegativeAxisIfNeed(in->shape(), conf.begin_params_axis());
  const Shape param_shape = Shape({in->shape().Count(begin_params_axis)});
  if (conf.center()) {
    BlobDesc* beta = GetBlobDesc4BnInOp("beta");
    beta->mut_shape() = param_shape;
    beta->set_data_type(in->data_type());
  }
  if (conf.scale()) {
    BlobDesc* gamma = GetBlobDesc4BnInOp("gamma");
    gamma->mut_shape() = param_shape;
    gamma->set_data_type(in->data_type());
    if (Global<JobDesc>::Get()->IsTrain()) { *GetBlobDesc4BnInOp("normalize_out") = *in; }
  }
  const int64_t begin_norm_axis = ShiftNegativeAxisIfNeed(in->shape(), conf.begin_norm_axis());
  const Shape bn_param_shape = Shape({in->shape().Count(0, begin_norm_axis)});
  BlobDesc* cudnn_bn_mean = GetBlobDesc4BnInOp("cudnn_bn_mean");
  cudnn_bn_mean->mut_shape() = bn_param_shape;
  cudnn_bn_mean->set_data_type(in->data_type());
  *GetBlobDesc4BnInOp("cudnn_bn_inv_variance") = *cudnn_bn_mean;
  *GetBlobDesc4BnInOp("cudnn_bn_scale_ones") = *cudnn_bn_mean;
  *GetBlobDesc4BnInOp("cudnn_bn_bias_zeros") = *cudnn_bn_mean;
}

void LayerNormOp::InferBwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK(parallel_ctx->policy() != kModelParallel);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const LayerNormOpConf& conf = op_conf().layer_norm_conf();
  if (conf.center() || conf.scale()) { *GetBlobDesc4BnInOp("reusable_bw_buf") = *in; }
  const int64_t begin_norm_axis = ShiftNegativeAxisIfNeed(in->shape(), conf.begin_norm_axis());
  const Shape bn_param_shape = Shape({in->shape().Count(0, begin_norm_axis)});
  BlobDesc* bn_scale_diff = GetBlobDesc4BnInOp("cudnn_bn_scale_diff_buf");
  bn_scale_diff->mut_shape() = bn_param_shape;
  bn_scale_diff->set_data_type(in->data_type());
  *GetBlobDesc4BnInOp("cudnn_bn_bias_diff_buf") = *bn_scale_diff;
}

REGISTER_OP(OperatorConf::kLayerNormConf, LayerNormOp);

}  // namespace oneflow
