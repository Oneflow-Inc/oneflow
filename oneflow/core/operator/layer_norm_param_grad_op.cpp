#include "oneflow/core/operator/layer_norm_param_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void LayerNormParamGradOp::InitFromOpConf() {
  CHECK(op_conf().has_layer_norm_param_grad_conf());
  const LayerNormParamGradOpConf& conf = op_conf().layer_norm_param_grad_conf();
  CHECK(conf.has_beta_diff() || conf.has_gamma_diff() || conf.has_normalized_diff());
  EnrollInputBn("dy", false);
  if (conf.has_beta_diff()) { EnrollOutputBn("beta_diff", false); }
  if (conf.has_gamma_diff()) {
    EnrollInputBn("normalized", false);
    EnrollOutputBn("gamma_diff", false);
  }
  if (conf.has_beta_diff() || conf.has_gamma_diff()) { EnrollTmpBn("reduce_buf"); }
  if (conf.has_normalized_diff()) { EnrollOutputBn("normalized_diff", false); }
  if (conf.has_normalized_diff() || conf.has_gamma_diff()) { CHECK(conf.has_gamma()); }
  if (conf.has_gamma()) { EnrollInputBn("gamma", false); }
}

Maybe<void> LayerNormParamGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_OR_RETURN(parallel_ctx->policy() != kModelParallel);
  const LayerNormParamGradOpConf& conf = op_conf().layer_norm_param_grad_conf();
  const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
  if (conf.has_beta_diff() || conf.has_gamma_diff()) {
    BlobDesc* reduce_buf = GetBlobDesc4BnInOp("reduce_buf");
    reduce_buf->set_data_type(dy->data_type());
    reduce_buf->mut_shape() = dy->shape();
  }
  const int64_t begin_params_axis = conf.begin_params_axis() < 0
                                        ? dy->shape().NumAxes() + conf.begin_params_axis()
                                        : conf.begin_params_axis();
  CHECK_GE_OR_RETURN(begin_params_axis, 1);
  CHECK_LT_OR_RETURN(begin_params_axis, dy->shape().NumAxes());
  std::vector<int64_t> param_shape_dim_vec;
  param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                             dy->shape().dim_vec().cbegin() + begin_params_axis,
                             dy->shape().dim_vec().cend());
  if (param_shape_dim_vec.empty()) { param_shape_dim_vec.push_back(1); }
  const Shape param_shape(param_shape_dim_vec);
  if (conf.has_beta_diff()) {
    BlobDesc* beta_diff = GetBlobDesc4BnInOp("beta_diff");
    beta_diff->mut_shape() = param_shape;
    beta_diff->set_data_type(dy->data_type());
  }
  if (conf.has_gamma_diff()) {
    const BlobDesc* normalized = GetBlobDesc4BnInOp("normalized");
    CHECK_EQ_OR_RETURN(normalized->data_type(), dy->data_type());
    CHECK_EQ_OR_RETURN(normalized->shape(), dy->shape());
    BlobDesc* gamma_diff = GetBlobDesc4BnInOp("gamma_diff");
    gamma_diff->mut_shape() = param_shape;
    gamma_diff->set_data_type(dy->data_type());
  }
  if (conf.has_normalized_diff()) { *GetBlobDesc4BnInOp("normalized_diff") = *dy; }
  if (conf.has_gamma()) {
    const BlobDesc* gamma = GetBlobDesc4BnInOp("gamma");
    CHECK_EQ_OR_RETURN(gamma->data_type(), dy->data_type());
    CHECK_EQ_OR_RETURN(gamma->shape(), param_shape);
  }
  return Maybe<void>::Ok();
}

Maybe<void> LayerNormParamGradOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  for (const auto& obn : output_bns()) { *HasBatchDim4BnInOp(obn) = false; }
  const LayerNormParamGradOpConf& conf = op_conf().layer_norm_param_grad_conf();
  if (conf.has_normalized_diff()) {
    *HasBatchDim4BnInOp("normalized_diff") = *HasBatchDim4BnInOp("dy");
  }
  return Maybe<void>::Ok();
}

void LayerNormParamGradOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Broadcast("gamma")
      .Split(output_bns(), 0)
      .PartialSum({"gamma_diff", "beta_diff"})
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kLayerNormParamGradConf, LayerNormParamGradOp);

}  // namespace oneflow
