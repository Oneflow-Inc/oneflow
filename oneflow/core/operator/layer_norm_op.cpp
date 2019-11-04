#include "oneflow/core/operator/layer_norm_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

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
  if (!(conf.center() || conf.scale())) { mut_op_conf()->set_trainable(false); }
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (conf.center()) {
    CHECK(conf.has_beta());
    EnrollInputBn("beta");
  }
  if (conf.scale()) {
    CHECK(conf.has_gamma());
    EnrollInputBn("gamma");
    EnrollOutputBn("normalized", false);
  }
  EnrollOutputBn("mean", false);
  EnrollOutputBn("inv_variance", false);
  EnrollConstBufBn("cudnn_bn_scale_ones");
  EnrollConstBufBn("cudnn_bn_bias_zeros");
}

Maybe<void> LayerNormOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in;
  const LayerNormOpConf& conf = op_conf().layer_norm_conf();
  const int64_t begin_params_axis = ShiftNegativeAxisIfNeed(in->shape(), conf.begin_params_axis());
  std::vector<int64_t> param_shape_dim_vec;
  param_shape_dim_vec.insert(param_shape_dim_vec.end(),
                             in->shape().dim_vec().cbegin() + begin_params_axis,
                             in->shape().dim_vec().cend());
  if (param_shape_dim_vec.empty()) { param_shape_dim_vec.push_back(1); }
  const Shape param_shape(param_shape_dim_vec);
  if (conf.center()) {
    CHECK_OR_RETURN(parallel_ctx->parallel_num() == 1
                    || sbp_signature->bn_in_op2sbp_parallel().at("beta").has_broadcast_parallel());
    const BlobDesc* beta = GetBlobDesc4BnInOp("beta");
    CHECK_EQ_OR_RETURN(beta->shape(), param_shape);
    CHECK_EQ_OR_RETURN(beta->data_type(), in->data_type());
  }
  if (conf.scale()) {
    CHECK_OR_RETURN(parallel_ctx->parallel_num() == 1
                    || sbp_signature->bn_in_op2sbp_parallel().at("gamma").has_broadcast_parallel());
    const BlobDesc* gamma = GetBlobDesc4BnInOp("gamma");
    CHECK_EQ_OR_RETURN(gamma->shape(), param_shape);
    CHECK_EQ_OR_RETURN(gamma->data_type(), in->data_type());
    *GetBlobDesc4BnInOp("normalized") = *in;
  }
  const int64_t begin_norm_axis = ShiftNegativeAxisIfNeed(in->shape(), conf.begin_norm_axis());
  std::vector<int64_t> bn_param_shape_dim_vec;
  bn_param_shape_dim_vec.insert(bn_param_shape_dim_vec.end(), in->shape().dim_vec().cbegin(),
                                in->shape().dim_vec().cbegin() + begin_norm_axis);
  const Shape bn_param_shape(bn_param_shape_dim_vec);
  BlobDesc* cudnn_bn_mean = GetBlobDesc4BnInOp("mean");
  cudnn_bn_mean->mut_shape() = bn_param_shape;
  DataType data_type = in->data_type() == DataType::kFloat16 ? DataType::kFloat : in->data_type();
  cudnn_bn_mean->set_data_type(data_type);
  *GetBlobDesc4BnInOp("inv_variance") = *cudnn_bn_mean;
  *GetBlobDesc4BnInOp("cudnn_bn_scale_ones") = *cudnn_bn_mean;
  *GetBlobDesc4BnInOp("cudnn_bn_bias_zeros") = *cudnn_bn_mean;
  return Maybe<void>::Ok();
}

Maybe<void> LayerNormOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp("in"); }
  return Maybe<void>::Ok();
}

Maybe<void> LayerNormOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Broadcast({"gamma", "beta"})
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kLayerNormConf, LayerNormOp);

}  // namespace oneflow
