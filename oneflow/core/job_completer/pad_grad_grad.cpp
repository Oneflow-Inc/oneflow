#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_pad_grad_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_grad");
    PadOpConf* conf = op_conf.mutable_pad_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_out("out");
    conf->set_constant_value(op.op_conf().pad_grad_conf().constant_value());
    *conf->mutable_paddings() = GetPbRfFromPbMessage<int32_t>(op.op_conf().pad_grad_conf(), "paddings");
    op_confs->push_back(op_conf);
    DiffLbi4BnInOp("in")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kPadGradConf, &GenerateBackwardOpConf);

}  // namespace oneflow
