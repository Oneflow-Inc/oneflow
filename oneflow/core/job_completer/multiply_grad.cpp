#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_multiply_conf());
  if (DiffLbi4BnInOp("in_0") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_in0_grad");
    MultiplyOpConf* conf = op_conf.mutable_multiply_conf();
    conf->set_out("out");

    conf->set_in_0(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("in_1")));

    op_confs->push_back(op_conf);

    DiffLbi4BnInOp("in_0")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in_0")->set_blob_name("out");
  }
  if (DiffLbi4BnInOp("in_1") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_in1_grad");
    MultiplyOpConf* conf = op_conf.mutable_multiply_conf();
    conf->set_out("out");

    conf->set_in_0(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("in_0")));

    op_confs->push_back(op_conf);

    DiffLbi4BnInOp("in_1")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in_1")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kMultiplyConf, &GenerateBackwardOpConf);

}  // namespace oneflow
