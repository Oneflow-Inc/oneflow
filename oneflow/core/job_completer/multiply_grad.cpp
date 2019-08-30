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
    MultiplyGradOpConf* conf = op_conf.mutable_multiply_grad_conf();

    conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_in_0(GenLogicalBlobName(op.BnInOp2Lbi("in_0"))); 
    conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("in_1")));
    conf->set_in_0_diff("in_0_diff");
    conf->set_in_1_diff("in_1_diff");

    op_confs->push_back(op_conf);

    DiffLbi4BnInOp("in_0")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in_0")->set_blob_name("in_0_diff");
  }
  if (DiffLbi4BnInOp("in_1") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_in1_grad");
    MultiplyGradOpConf* conf = op_conf.mutable_multiply_grad_conf();

    conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_in_0(GenLogicalBlobName(op.BnInOp2Lbi("in_0"))); 
    conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("in_1")));
    conf->set_in_0_diff("in_0_diff");
    conf->set_in_1_diff("in_1_diff");

    op_confs->push_back(op_conf);

    DiffLbi4BnInOp("in_1")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("in_1")->set_blob_name("in_1_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kMultiplyConf, &GenerateBackwardOpConf);

}  // namespace oneflow
