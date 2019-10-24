#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_exp_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_grad");
    MultiplyOpConf* conf = op_conf.mutable_multiply_conf();
    conf->set_out("out");
    conf->set_in_0(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    op_confs->push_back(op_conf);
    DiffLbi4BnInOp("in")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kExpConf, &GenerateBackwardOpConf);

}  // namespace oneflow
