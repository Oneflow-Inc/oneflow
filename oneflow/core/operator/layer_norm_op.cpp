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
  if (!(conf.center() || conf.scale())) { mut_op_conf()->set_trainable(false); }
  const bool fw_bw_split =
      Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf();
  if (!fw_bw_split) {
    CHECK(!conf.has_beta());
    CHECK(!conf.has_gamma());
  }
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (conf.center()) {
    if (fw_bw_split && conf.has_beta()) {
      EnrollInputBn("beta");
    } else {
      EnrollModelBn("beta");
    }
  }
  if (conf.scale()) {
    if (fw_bw_split && conf.has_gamma()) {
      EnrollInputBn("gamma");
      EnrollOutputBn("normalized", false);
    } else {
      EnrollModelBn("gamma");
      EnrollDataTmpBn("normalized");
    }
  }
  if (fw_bw_split) {
    EnrollOutputBn("mean", false);
    EnrollOutputBn("inv_variance", false);
  } else {
    EnrollDataTmpBn("mean");
    EnrollDataTmpBn("inv_variance");
  }
  EnrollConstBufBn("cudnn_bn_scale_ones");
  EnrollConstBufBn("cudnn_bn_bias_zeros");
  EnrollBwBufBn("cudnn_bn_scale_diff_buf");
  EnrollBwBufBn("cudnn_bn_bias_diff_buf");
  if (op_conf().trainable()) { EnrollBwBufBn("bw_reduce_buf"); }
}

void LayerNormOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  CHECK(parallel_ctx->policy() != kModelParallel);
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
    if (conf.has_beta()) {
      const BlobDesc* beta = GetBlobDesc4BnInOp("beta");
      CHECK_EQ(beta->shape(), param_shape);
      CHECK_EQ(beta->data_type(), in->data_type());
    } else {
      BlobDesc* beta = GetBlobDesc4BnInOp("beta");
      beta->mut_shape() = param_shape;
      beta->set_data_type(in->data_type());
    }
  }
  if (conf.scale()) {
    if (conf.has_gamma()) {
      const BlobDesc* gamma = GetBlobDesc4BnInOp("gamma");
      CHECK_EQ(gamma->shape(), param_shape);
      CHECK_EQ(gamma->data_type(), in->data_type());
    } else {
      BlobDesc* gamma = GetBlobDesc4BnInOp("gamma");
      gamma->mut_shape() = param_shape;
      gamma->set_data_type(in->data_type());
    }
    *GetBlobDesc4BnInOp("normalized") = *in;
  }
  const int64_t begin_norm_axis = ShiftNegativeAxisIfNeed(in->shape(), conf.begin_norm_axis());
  std::vector<int64_t> bn_param_shape_dim_vec;
  bn_param_shape_dim_vec.insert(bn_param_shape_dim_vec.end(), in->shape().dim_vec().cbegin(),
                                in->shape().dim_vec().cbegin() + begin_norm_axis);
  const Shape bn_param_shape(bn_param_shape_dim_vec);
  BlobDesc* cudnn_bn_mean = GetBlobDesc4BnInOp("mean");
  cudnn_bn_mean->mut_shape() = bn_param_shape;
  cudnn_bn_mean->set_data_type(in->data_type());
  *GetBlobDesc4BnInOp("inv_variance") = *cudnn_bn_mean;
  *GetBlobDesc4BnInOp("cudnn_bn_scale_ones") = *cudnn_bn_mean;
  *GetBlobDesc4BnInOp("cudnn_bn_bias_zeros") = *cudnn_bn_mean;
}

void LayerNormOp::InferBwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK(parallel_ctx->policy() != kModelParallel);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const LayerNormOpConf& conf = op_conf().layer_norm_conf();
  if (op_conf().trainable()) { *GetBlobDesc4BnInOp("bw_reduce_buf") = *in; }
  const int64_t begin_norm_axis = ShiftNegativeAxisIfNeed(in->shape(), conf.begin_norm_axis());
  const Shape bn_param_shape = Shape({in->shape().Count(0, begin_norm_axis)});
  BlobDesc* bn_scale_diff = GetBlobDesc4BnInOp("cudnn_bn_scale_diff_buf");
  bn_scale_diff->mut_shape() = bn_param_shape;
  bn_scale_diff->set_data_type(in->data_type());
  *GetBlobDesc4BnInOp("cudnn_bn_bias_diff_buf") = *bn_scale_diff;
}

bool LayerNormOp::IsInputBlobAllowedModelSplit(const std::string& ibn) const {
  return ibn == "beta" || ibn == "gamma";
}

void LayerNormOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  const LayerNormOpConf& conf = op_conf().layer_norm_conf();
  if (conf.has_beta() || conf.has_gamma()) {
    op_parallel_signatures->emplace_back(Make_DS_MB_2_DS_OpParallelSignature(this));
  } else {
    op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
  }
}

REGISTER_OP(OperatorConf::kLayerNormConf, LayerNormOp);

}  // namespace oneflow
