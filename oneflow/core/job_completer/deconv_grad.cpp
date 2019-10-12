#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

// auto GetNdConvOpConf(OperatorConf& data_grad_op, int32_t Ndims){
//   if (Ndims == 1){
//     Conv1DOpConf* conf = data_grad_op.mutable_conv_1d_conf();
//     return conf;
//   }else if(Ndims == 2){
//     Conv2DOpConf* conf = data_grad_op.mutable_conv_2d_conf();
//     return conf;
//   }else if(Ndims == 3){
//     Conv3DOpConf* conf = data_grad_op.mutable_conv_3d_conf();
//     return conf;
//   }else{
//     UNIMPLEMENTED();
//   }
// }

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_deconv_conf());
  const ConvConf& conv_conf = op.op_conf().deconv_conf().conv_conf();
  const std::string out_diff_lbn = GenLogicalBlobName(*DiffLbi4BnInOp("y"));
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
  LogicalBlobId* filter_diff_lbi = DiffLbi4BnInOp("filter");
  if (filter_diff_lbi != nullptr) {
    OperatorConf filter_grad_op;
    filter_grad_op.set_name("System-AutoGrad-" + op.op_name() + "-FilterGrad");
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

    // if (Ndims == 1){
    //   Conv1DOpConf* conf = data_grad_op.mutable_conv_1d_conf();
    // }else if (Ndims == 2){
    //   Conv2DOpConf* conf = data_grad_op.mutable_conv_2d_conf();
    // }else if (Ndims == 3){
    //   Conv3DOpConf* conf = data_grad_op.mutable_conv_3d_conf();
    // }else{
    //   UNIMPLEMENTED();
    // }

    if (Ndims == 1) {
      Conv1DOpConf* conf = data_grad_op.mutable_conv_1d_conf();
      conf->set_in(out_diff_lbn);
      conf->set_weight(GenLogicalBlobName(op.BnInOp2Lbi("filter")));
      conf->set_out("y");
      conf->set_padding(GetValFromPbMessage<std::string>(conv_conf, "padding"));
      conf->set_data_format(GetValFromPbMessage<std::string>(conv_conf, "data_format"));

      const BlobDesc& filter_logical_blob_desc = LogicalBlobDesc4BnInOp("filter");
      conf->set_filters(filter_logical_blob_desc.shape().At(0));

      *conf->mutable_kernel_size() = GetPbRfFromPbMessage<int32_t>(conv_conf, "kernel_size");
      *conf->mutable_strides() = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
      *conf->mutable_dilation_rate() = GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
      conf->set_use_bias(false);
      op_confs->push_back(data_grad_op);
      in_diff_lbi->set_op_name(data_grad_op.name());
      in_diff_lbi->set_blob_name(conf->out());
    } else if (Ndims == 2) {
      Conv2DOpConf* conf = data_grad_op.mutable_conv_2d_conf();
      conf->set_in(out_diff_lbn);
      conf->set_weight(GenLogicalBlobName(op.BnInOp2Lbi("filter")));
      conf->set_out("y");
      conf->set_padding(GetValFromPbMessage<std::string>(conv_conf, "padding"));
      conf->set_data_format(GetValFromPbMessage<std::string>(conv_conf, "data_format"));

      const BlobDesc& filter_logical_blob_desc = LogicalBlobDesc4BnInOp("filter");
      conf->set_filters(filter_logical_blob_desc.shape().At(0));

      *conf->mutable_kernel_size() = GetPbRfFromPbMessage<int32_t>(conv_conf, "kernel_size");
      *conf->mutable_strides() = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
      *conf->mutable_dilation_rate() = GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
      conf->set_use_bias(false);
      op_confs->push_back(data_grad_op);
      in_diff_lbi->set_op_name(data_grad_op.name());
      in_diff_lbi->set_blob_name(conf->out());
    } else if (Ndims == 3) {
      Conv3DOpConf* conf = data_grad_op.mutable_conv_3d_conf();
      conf->set_in(out_diff_lbn);
      conf->set_weight(GenLogicalBlobName(op.BnInOp2Lbi("filter")));
      conf->set_out("y");
      conf->set_padding(GetValFromPbMessage<std::string>(conv_conf, "padding"));
      conf->set_data_format(GetValFromPbMessage<std::string>(conv_conf, "data_format"));

      const BlobDesc& filter_logical_blob_desc = LogicalBlobDesc4BnInOp("filter");
      conf->set_filters(filter_logical_blob_desc.shape().At(0));

      *conf->mutable_kernel_size() = GetPbRfFromPbMessage<int32_t>(conv_conf, "kernel_size");
      *conf->mutable_strides() = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
      *conf->mutable_dilation_rate() = GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
      conf->set_use_bias(false);
      op_confs->push_back(data_grad_op);
      in_diff_lbi->set_op_name(data_grad_op.name());
      in_diff_lbi->set_blob_name(conf->out());
    } else {
      UNIMPLEMENTED();
    }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kDeconvConf, &GenerateBackwardOpConf);

}  // namespace oneflow
