#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf4ScalarAdd(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_scalar_add_conf());
  if (DiffLbi4BnInOp("in") != nullptr) { *DiffLbi4BnInOp("in") = *DiffLbi4BnInOp("out"); }
}

void GenerateBackwardOpConf4ScalarMul(
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

void GenerateBackwardOpConf4ScalarPow(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_scalar_pow_conf());
  int32_t pow_operand = op.op_conf().scalar_pow_conf().int_operand();
  if (DiffLbi4BnInOp("in") != nullptr) {
    std::string tmp_lbn = GenLogicalBlobName(op.BnInOp2Lbi("in"));
    if (pow_operand > 2) {
      OperatorConf grad_scalar_pow_op;
      grad_scalar_pow_op.set_name(op.op_name() + "_grad_scalar_pow");
      ScalarPowOpConf* conf = grad_scalar_pow_op.mutable_scalar_pow_conf();
      conf->set_int_operand(pow_operand - 1);
      conf->set_in(tmp_lbn);
      conf->set_out("out");
      op_confs->push_back(grad_scalar_pow_op);
      tmp_lbn = grad_scalar_pow_op.name() + "/out";
    }
    {
      OperatorConf grad_scalar_mul_op;
      grad_scalar_mul_op.set_name(op.op_name() + "_grad_scalar_mul");
      ScalarMulOpConf* conf = grad_scalar_mul_op.mutable_scalar_mul_conf();
      conf->set_int_operand(pow_operand);
      conf->set_out("out");
      conf->set_in(tmp_lbn);
      op_confs->push_back(grad_scalar_mul_op);
      tmp_lbn = grad_scalar_mul_op.name() + "/out";
    }
    {
      OperatorConf grad_broadcast_mul_op;
      grad_broadcast_mul_op.set_name(op.op_name() + "_grad_broadcast_mul");
      BroadcastMulOpConf* conf = grad_broadcast_mul_op.mutable_broadcast_mul_conf();
      conf->set_a(tmp_lbn);
      conf->set_b(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      op_confs->push_back(grad_broadcast_mul_op);
      DiffLbi4BnInOp("in")->set_op_name(grad_broadcast_mul_op.name());
      DiffLbi4BnInOp("in")->set_blob_name("out");
    }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarAddConf, &GenerateBackwardOpConf4ScalarAdd);
REGISTER_OP_GRAD(OperatorConf::kScalarMulConf, &GenerateBackwardOpConf4ScalarMul);
REGISTER_OP_GRAD(OperatorConf::kScalarMulConf, &GenerateBackwardOpConf4ScalarPow);

}  // namespace oneflow
