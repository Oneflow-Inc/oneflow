#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_prelu_conf());
  const auto& conf = op.op_conf().prelu_conf();
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf prelu_data_grad_op;
    prelu_data_grad_op.set_name(op.op_name() + "_data_grad");
    PReluDataGradOpConf* prelu_data_grad_op_conf =
        prelu_data_grad_op.mutable_prelu_data_grad_conf();
    prelu_data_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    prelu_data_grad_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    prelu_data_grad_op_conf->set_alpha(GenLogicalBlobName(op.BnInOp2Lbi("alpha")));
    prelu_data_grad_op_conf->set_data_format(conf.data_format());
    prelu_data_grad_op_conf->set_channel_shared(conf.channel_shared());
    prelu_data_grad_op_conf->set_dx("dx");
    op_confs->push_back(prelu_data_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(prelu_data_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
  if (DiffLbi4BnInOp("alpha") != nullptr) {
    OperatorConf prelu_alpha_grad_op;
    prelu_alpha_grad_op.set_name(op.op_name() + "_alpha_grad");
    PReluAlphaGradOpConf* prelu_alpha_grad_op_conf =
        prelu_alpha_grad_op.mutable_prelu_alpha_grad_conf();
    prelu_alpha_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    prelu_alpha_grad_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    prelu_alpha_grad_op_conf->set_data_format(conf.data_format());
    prelu_alpha_grad_op_conf->set_channel_shared(conf.channel_shared());
    prelu_alpha_grad_op_conf->set_alpha_grad("alpha_grad");
    op_confs->push_back(prelu_alpha_grad_op);
    DiffLbi4BnInOp("alpha")->set_op_name(prelu_alpha_grad_op.name());
    DiffLbi4BnInOp("alpha")->set_blob_name("alpha_grad");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kPreluConf, &GenerateBackwardOpConf);

}  // namespace oneflow
