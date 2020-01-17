#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sqrt_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf broadcast_div_op;
    broadcast_div_op.set_name(op.op_name() + "_broadcast_div_grad");
    BroadcastDivOpConf* broadcast_div_op_conf = broadcast_div_op.mutable_broadcast_div_conf();
    broadcast_div_op_conf->set_a(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    broadcast_div_op_conf->set_b(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    broadcast_div_op_conf->set_out("out");
    op_confs->push_back(broadcast_div_op);
    OperatorConf scalar_mul_op;
    scalar_mul_op.set_name(op.op_name() + "_scalar_mul_in_grad");
    ScalarMulOpConf* scalar_mul_op_conf = scalar_mul_op.mutable_scalar_mul_conf();
    scalar_mul_op_conf->set_float_operand(0.5);
    scalar_mul_op_conf->set_in(broadcast_div_op.name() + "/out");
    scalar_mul_op_conf->set_out("out");
    op_confs->push_back(scalar_mul_op);
    DiffLbi4BnInOp("in")->set_op_name(scalar_mul_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(scalar_mul_op_conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSqrtConf, &GenerateBackwardOpConf);

}  // namespace oneflow
