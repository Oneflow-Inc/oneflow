#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

ConvConf ConvConfFromConvOpConf(const OperatorConf& op_conf) {
  int32_t n_spatial_dims = 0;
  const PbMessage* msg = nullptr;
  if (op_conf.has_conv_1d_conf()) {
    n_spatial_dims = 1;
    msg = &op_conf.conv_1d_conf();
  } else if (op_conf.has_conv_2d_conf()) {
    n_spatial_dims = 2;
    msg = &op_conf.conv_2d_conf();
  } else if (op_conf.has_conv_3d_conf()) {
    n_spatial_dims = 3;
    msg = &op_conf.conv_3d_conf();
  } else {
    UNIMPLEMENTED();
  }
  ConvConf conv_conf;
  conv_conf.set_num_spatial_dims(n_spatial_dims);
  conv_conf.set_data_format(GetValFromPbMessage<std::string>(*msg, "data_format"));
  conv_conf.set_padding(GetValFromPbMessage<std::string>(*msg, "padding"));
  *conv_conf.mutable_kernel_size() = GetPbRfFromPbMessage<int32_t>(*msg, "kernel_size");
  *conv_conf.mutable_strides() = GetPbRfFromPbMessage<int32_t>(*msg, "strides");
  *conv_conf.mutable_dilation_rate() = GetPbRfFromPbMessage<int32_t>(*msg, "dilation_rate");
  conv_conf.set_group_num(GetValFromPbMessage<int32_t>(*msg, "group_num"));
  return conv_conf;
}

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_conv_1d_conf() || op.op_conf().has_conv_2d_conf()
        || op.op_conf().has_conv_3d_conf());
  const ConvConf conv_conf = ConvConfFromConvOpConf(op.op_conf());
  const std::string out_diff_lbn = GenLogicalBlobName(*DiffLbi4BnInOp("out"));
  if (op.GetValFromCustomizedConf<bool>("use_bias")) {
    LogicalBlobId* bias_diff_lbi = DiffLbi4BnInOp("bias");
    if (bias_diff_lbi != nullptr) {
      OperatorConf bias_grad_op;
      bias_grad_op.set_name("System-AutoGrad-" + op.op_name() + "-BiasGrad");
      ConvBiasGradOpConf* conf = bias_grad_op.mutable_conv_bias_grad_conf();
      conf->set_data_format(conv_conf.data_format());
      conf->set_num_spatial_dims(conv_conf.num_spatial_dims());
      conf->set_dy(out_diff_lbn);
      conf->set_bias_diff("bias_diff");
      op_confs->push_back(bias_grad_op);
      bias_diff_lbi->set_op_name(bias_grad_op.name());
      bias_diff_lbi->set_blob_name(conf->bias_diff());
    }
  }
  LogicalBlobId* filter_diff_lbi = DiffLbi4BnInOp("weight");
  if (filter_diff_lbi != nullptr) {
    OperatorConf filter_grad_op;
    filter_grad_op.set_name("System-AutoGrad-" + op.op_name() + "-FilterGrad");
    ConvFilterGradOpConf* conf = filter_grad_op.mutable_conv_filter_grad_conf();
    *conf->mutable_conv_conf() = conv_conf;
    conf->set_dy(out_diff_lbn);
    conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    conf->set_filter_diff("filter_diff");
    op_confs->push_back(filter_grad_op);
    filter_diff_lbi->set_op_name(filter_grad_op.name());
    filter_diff_lbi->set_blob_name(conf->filter_diff());
  }
  LogicalBlobId* in_diff_lbi = DiffLbi4BnInOp("in");
  if (in_diff_lbi != nullptr) {
    OperatorConf data_grad_op;
    data_grad_op.set_name("System-AutoGrad-" + op.op_name() + "-DataGrad");
    ConvDataGradOpConf* conf = data_grad_op.mutable_conv_data_grad_conf();
    *conf->mutable_conv_conf() = conv_conf;
    conf->set_dy(out_diff_lbn);
    conf->set_filter(GenLogicalBlobName(op.BnInOp2Lbi("weight")));
    conf->set_x_like(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    conf->set_dx("dx");
    op_confs->push_back(data_grad_op);
    in_diff_lbi->set_op_name(data_grad_op.name());
    in_diff_lbi->set_blob_name(conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kConv1DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kConv2DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kConv3DConf, &GenerateBackwardOpConf);

}  // namespace oneflow
