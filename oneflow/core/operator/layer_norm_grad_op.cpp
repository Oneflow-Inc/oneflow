#include "oneflow/core/operator/layer_norm_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void LayerNormGradOp::InitFromOpConf() {
  CHECK(op_conf().has_layer_norm_grad_conf());
  const LayerNormGradOpConf& conf = op_conf().layer_norm_grad_conf();
  EnrollInputBn("dy", false);
  EnrollInputBn("x", false);
  CHECK_EQ(conf.has_mean(), conf.has_inv_variance());
  if (conf.has_mean() && conf.has_inv_variance()) {
    EnrollInputBn("mean", false);
    EnrollInputBn("inv_variance", false);
  }
  EnrollOutputBn("dx", false);
  EnrollConstBufBn("cudnn_bn_scale_ones");
  EnrollTmpBn("cudnn_bn_scale_diff_buf");
  EnrollTmpBn("cudnn_bn_bias_diff_buf");
}

void LayerNormGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK(parallel_ctx->policy() != kModelParallel);
  const LayerNormGradOpConf& conf = op_conf().layer_norm_grad_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  CHECK_GE(conf.begin_norm_axis(), 1);
  CHECK_LT(conf.begin_norm_axis(), dy->shape().NumAxes());
  const BlobDesc* x = GetBlobDesc4BnInOp("x");
  CHECK_EQ(dy->data_type(), x->data_type());
  CHECK_EQ(dy->shape(), x->shape());
  const int64_t begin_norm_axis = conf.begin_norm_axis() < 0
                                      ? dy->shape().NumAxes() + conf.begin_norm_axis()
                                      : conf.begin_norm_axis();
  CHECK_GE(begin_norm_axis, 1);
  CHECK_LT(begin_norm_axis, x->shape().NumAxes());
  std::vector<int64_t> bn_param_shape_dim_vec;
  bn_param_shape_dim_vec.insert(bn_param_shape_dim_vec.end(), x->shape().dim_vec().cbegin(),
                                x->shape().dim_vec().cbegin() + begin_norm_axis);
  const Shape bn_param_shape(bn_param_shape_dim_vec);
  const BlobDesc* mean = GetBlobDesc4BnInOp("mean");
  const BlobDesc* inv_variance = GetBlobDesc4BnInOp("inv_variance");
  if (mean || inv_variance) {
    CHECK_NOTNULL(mean);
    CHECK_NOTNULL(inv_variance);
    // CHECK_EQ(mean->data_type(), x->data_type());
    CHECK_EQ(mean->shape(), bn_param_shape);
    // CHECK_EQ(inv_variance->data_type(), x->data_type());
    CHECK_EQ(inv_variance->shape(), bn_param_shape);
  }
  BlobDesc* dx = GetBlobDesc4BnInOp("dx");
  BlobDesc* bn_scale = GetBlobDesc4BnInOp("cudnn_bn_scale_ones");
  BlobDesc* bn_scale_diff = GetBlobDesc4BnInOp("cudnn_bn_scale_diff_buf");
  BlobDesc* bn_bias_diff = GetBlobDesc4BnInOp("cudnn_bn_bias_diff_buf");
  *dx = *dy;
  bn_scale->mut_shape() = Shape({dy->shape().Count(0, begin_norm_axis)});
  DataType data_type = dy->data_type() == DataType::kFloat16 ? DataType::kFloat : dy->data_type();
  bn_scale->set_data_type(data_type);
  *bn_scale_diff = *bn_scale;
  *bn_bias_diff = *bn_scale;
}

void LayerNormGradOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kLayerNormGradConf, LayerNormGradOp);

}  // namespace oneflow
