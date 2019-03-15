#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_scalar_mul_conf());
  const ScalarMulOpConf& conf = op.op_conf().scalar_mul_conf();
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf reverse_scalar_mul_op;
    reverse_scalar_mul_op.set_name(op.op_name() + "_grad");
    ScalarMulOpConf* reverse_scalar_mul_op_conf = reverse_scalar_mul_op.mutable_scalar_mul_conf();
    if (conf.has_int_operand()) {
      reverse_scalar_mul_op_conf->set_int_operand(conf.int_operand());
    } else if (conf.has_float_operand()) {
      reverse_scalar_mul_op_conf->set_float_operand(conf.float_operand());
    } else {
      UNIMPLEMENTED();
    }
    reverse_scalar_mul_op_conf->set_out("out");
    reverse_scalar_mul_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    op_confs->push_back(reverse_scalar_mul_op);
    DiffLbi4BnInOp("in")->set_op_name(reverse_scalar_mul_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarMulConf, &GenerateBackwardOpConf);

}  // namespace oneflow
