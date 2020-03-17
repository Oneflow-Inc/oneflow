#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

template<typename ConvNdOpConf>
void SetNdConvOpConf(
    ConvNdOpConf* conf, const ConvConf& conv_conf, const Operator& op,
    const std::string out_diff_lbn, LogicalBlobId* in_diff_lbi, OperatorConf& data_grad_op,
    std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  conf->set_in(out_diff_lbn);
  conf->set_weight(GenLogicalBlobName(op.BnInOp2Lbi("weight")));
  conf->set_out("y");
  conf->set_padding(GetValFromPbMessage<std::string>(conv_conf, "padding"));
  conf->set_data_format(GetValFromPbMessage<std::string>(conv_conf, "data_format"));

  const BlobDesc& weight_logical_blob_desc = LogicalBlobDesc4BnInOp("weight");
  conf->set_filters(weight_logical_blob_desc.shape().At(0));

  *conf->mutable_kernel_size() = GetPbRfFromPbMessage<int32_t>(conv_conf, "kernel_size");
  *conf->mutable_strides() = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
  *conf->mutable_dilation_rate() = GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
  CHECK(conv_conf.has_torch_style_padding_conf());
  *conf->mutable_torch_style_padding_conf()->mutable_padding_needed() =
      GetPbRfFromPbMessage<int32_t>(conv_conf.torch_style_padding_conf(), "padding_needed");
  conf->set_use_bias(false);
  op_confs->push_back(data_grad_op);
  in_diff_lbi->set_op_name(data_grad_op.name());
  in_diff_lbi->set_blob_name(conf->out());
}

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_deconv_conf());
  const ConvConf& conv_conf = op.op_conf().deconv_conf().conv_conf();
  const std::string out_diff_lbn = GenLogicalBlobName(*DiffLbi4BnInOp("y"));

  LogicalBlobId* filter_diff_lbi = DiffLbi4BnInOp("weight");
  if (filter_diff_lbi != nullptr) {
    OperatorConf filter_grad_op;
    filter_grad_op.set_name("System-AutoGrad-" + op.op_name() + "-weightGrad");
    ConvFilterGradOpConf* conf = filter_grad_op.mutable_conv_filter_grad_conf();
    *conf->mutable_conv_conf() = conv_conf;
    conf->set_dy(GenLogicalBlobName(op.BnInOp2Lbi("x")));
    conf->set_x(out_diff_lbn);
    conf->set_filter_diff("filter_diff");
    op_confs->push_back(filter_grad_op);
    filter_diff_lbi->set_op_name(filter_grad_op.name());
    filter_diff_lbi->set_blob_name(conf->filter_diff());
  }

  LogicalBlobId* in_diff_lbi = DiffLbi4BnInOp("x");
  if (in_diff_lbi != nullptr) {
    OperatorConf data_grad_op;
    data_grad_op.set_name("System-AutoGrad-" + op.op_name() + "-DataGrad");
    int32_t Ndims = conv_conf.num_spatial_dims();

    if (Ndims == 1) {
      Conv1DOpConf* conf = data_grad_op.mutable_conv_1d_conf();
      SetNdConvOpConf(conf, conv_conf, op, out_diff_lbn, in_diff_lbi, data_grad_op, op_confs,
                      LogicalBlobDesc4BnInOp);
    } else if (Ndims == 2) {
      Conv2DOpConf* conf = data_grad_op.mutable_conv_2d_conf();
      SetNdConvOpConf(conf, conv_conf, op, out_diff_lbn, in_diff_lbi, data_grad_op, op_confs,
                      LogicalBlobDesc4BnInOp);
    } else if (Ndims == 3) {
      Conv3DOpConf* conf = data_grad_op.mutable_conv_3d_conf();
      SetNdConvOpConf(conf, conv_conf, op, out_diff_lbn, in_diff_lbi, data_grad_op, op_confs,
                      LogicalBlobDesc4BnInOp);
    } else {
      UNIMPLEMENTED();
    }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kDeconvConf, &GenerateBackwardOpConf);

}  // namespace oneflow
