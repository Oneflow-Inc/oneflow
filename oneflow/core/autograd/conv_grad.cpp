#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

ConvConf GenConvConfFromConvOpConf(const OperatorConf& op_conf) {
  int32_t n_dims = 0;
  const PbMessage* msg = nullptr;
  if (op_conf.has_conv_1d_conf()) {
    n_dims = 1;
    msg = &op_conf.conv_1d_conf();
  } else if (op_conf.has_conv_2d_conf()) {
    n_dims = 2;
    msg = &op_conf.conv_2d_conf();
  } else if (op_conf.has_conv_3d_conf()) {
    n_dims = 3;
    msg = &op_conf.conv_3d_conf();
  } else {
    UNIMPLEMENTED();
  }
  ConvConf conv_conf;
  conv_conf.set_num_dims(n_dims);
  conv_conf.set_data_format(GetValFromPbMessage<std::string>(*msg, "data_format"));
  conv_conf.set_padding(GetValFromPbMessage<std::string>(*msg, "padding"));
  *conv_conf.mutable_kernel_size() = GetPbRfFromPbMessage<int32_t>(*msg, "kernel_size");
  *conv_conf.mutable_strides() = GetPbRfFromPbMessage<int32_t>(*msg, "strides");
  *conv_conf.mutable_dilation_rate() = GetPbRfFromPbMessage<int32_t>(*msg, "dilation_rate");
  return conv_conf;
}

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_conv_1d_conf() || op.op_conf().has_conv_2d_conf()
        || op.op_conf().has_conv_2d_conf());
  const ConvConf conv_conf = GenConvConfFromConvOpConf(op.op_conf());
  if (op.GetValFromCustomizedConf<bool>("use_bias") && DiffLbi4BnInOp("bias") != nullptr) {
    OperatorConf bias_grad_op;
    bias_grad_op.set_name(op.op_name() + "_bias_grad");
    ConvBiasGradOpConf* conf = bias_grad_op.mutable_conv_bias_grad_conf();
    conf->set_data_format(conv_conf.data_format());
    conf->set_num_dims(conv_conf.num_dims());
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_bias_diff("bias_diff");
    op_confs->push_back(bias_grad_op);
    DiffLbi4BnInOp("bias")->set_op_name(bias_grad_op.name());
    DiffLbi4BnInOp("bias")->set_blob_name("bias_diff");
  }
  if (DiffLbi4BnInOp("weight") != nullptr) {
    OperatorConf filter_grad_op;
    filter_grad_op.set_name(op.op_name() + "_filter_grad");
    ConvFilterGradOpConf* conf = filter_grad_op.mutable_conv_filter_grad_conf();
    *conf->mutable_conv_conf() = conv_conf;
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    conf->set_filter_diff("filter_diff");
    op_confs->push_back(filter_grad_op);
    DiffLbi4BnInOp("weight")->set_op_name(filter_grad_op.name());
    DiffLbi4BnInOp("weight")->set_blob_name("filter_diff");
  }
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf data_grad_op;
    data_grad_op.set_name(op.op_name() + "_data_grad");
    ConvDataGradOpConf* conf = data_grad_op.mutable_conv_data_grad_conf();
    *conf->mutable_conv_conf() = conv_conf;
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_filter(GenLogicalBlobName(op.BnInOp2Lbi("weight")));
    conf->set_x_like(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    conf->set_dy("dx");
    op_confs->push_back(data_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(data_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kConv1DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kConv2DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kConv3DConf, &GenerateBackwardOpConf);

}  // namespace oneflow
