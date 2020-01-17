#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_square_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf multiply_in_op;
    multiply_in_op.set_name(op.op_name() + "_multiply_in_grad");
    MultiplyOpConf* multiply_in_op_conf = multiply_in_op.mutable_multiply_conf();
    multiply_in_op_conf->set_in_0(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    multiply_in_op_conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    multiply_in_op_conf->set_out("out");
    op_confs->push_back(multiply_in_op);
    OperatorConf scalar_mul_op;
    scalar_mul_op.set_name(op.op_name() + "_scalar_mul_in_grad");
    ScalarMulOpConf* scalar_mul_op_conf = scalar_mul_op.mutable_scalar_mul_conf();
    scalar_mul_op_conf->set_float_operand(2);
    scalar_mul_op_conf->set_in(multiply_in_op.name() + "/out");
    scalar_mul_op_conf->set_out("out");
    op_confs->push_back(scalar_mul_op);
    DiffLbi4BnInOp("in")->set_op_name(scalar_mul_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(scalar_mul_op_conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSquareConf, &GenerateBackwardOpConf);

}  // namespace oneflow
