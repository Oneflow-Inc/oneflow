#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  auto* diff_lbi = DiffLbi4BnInOp("prediction");
  if (diff_lbi != nullptr) {
    OperatorConf mul_zero_op;
    mul_zero_op.set_name(op.op_name() + "_grad_stage0");
    ScalarMulOpConf* mul_zero_op_conf = mul_zero_op.mutable_scalar_mul_conf();
    mul_zero_op_conf->set_in(GenLogicalBlobName(op.BnInOp2Lbi("prediction")));
    mul_zero_op_conf->set_out("out");
    mul_zero_op_conf->set_int_operand(0);
    op_confs->push_back(mul_zero_op);

    OperatorConf add_one_op;
    add_one_op.set_name(op.op_name() + "_grad_stage1");
    ScalarAddOpConf* add_one_op_conf = add_one_op.mutable_scalar_add_conf();
    add_one_op_conf->set_in(mul_zero_op.name() + "/out");
    add_one_op_conf->set_out("out");
    add_one_op_conf->set_int_operand(1);
    op_confs->push_back(add_one_op);

    diff_lbi->set_op_name(add_one_op.name());
    diff_lbi->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kIdentityLossConf, &GenerateBackwardOpConf);

}  // namespace oneflow
