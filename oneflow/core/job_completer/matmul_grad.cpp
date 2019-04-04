#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_matmul_conf());
  const MatmulOpConf& conf = op.op_conf().matmul_conf();
  if (DiffLbi4BnInOp("a") != nullptr) {
    OperatorConf matmul_a_op;
    matmul_a_op.set_name(op.op_name() + "_grad_a");
    MatmulOpConf* matmul_a_op_conf = matmul_a_op.mutable_matmul_conf();
    matmul_a_op_conf->set_out("out");
    if (conf.transpose_a()) {
      matmul_a_op_conf->set_a(GenLogicalBlobName(op.BnInOp2Lbi("b")));
      matmul_a_op_conf->set_b(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      matmul_a_op_conf->set_transpose_a(conf.transpose_b());
      matmul_a_op_conf->set_transpose_b(true);
    } else {
      matmul_a_op_conf->set_a(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      matmul_a_op_conf->set_b(GenLogicalBlobName(op.BnInOp2Lbi("b")));
      matmul_a_op_conf->set_transpose_a(false);
      matmul_a_op_conf->set_transpose_b(!conf.transpose_b());
    }
    op_confs->push_back(matmul_a_op);
    DiffLbi4BnInOp("a")->set_op_name(matmul_a_op.name());
    DiffLbi4BnInOp("a")->set_blob_name("out");
  }
  if (DiffLbi4BnInOp("b") != nullptr) {
    OperatorConf matmul_b_op;
    matmul_b_op.set_name(op.op_name() + "_grad_b");
    MatmulOpConf* matmul_b_op_conf = matmul_b_op.mutable_matmul_conf();
    matmul_b_op_conf->set_out("out");
    if (conf.transpose_b()) {
      matmul_b_op_conf->set_a(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      matmul_b_op_conf->set_b(GenLogicalBlobName(op.BnInOp2Lbi("a")));
      matmul_b_op_conf->set_transpose_a(true);
      matmul_b_op_conf->set_transpose_b(conf.transpose_a());
    } else {
      matmul_b_op_conf->set_a(GenLogicalBlobName(op.BnInOp2Lbi("a")));
      matmul_b_op_conf->set_b(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
      matmul_b_op_conf->set_transpose_a(!conf.transpose_a());
      matmul_b_op_conf->set_transpose_b(false);
    }
    op_confs->push_back(matmul_b_op);
    DiffLbi4BnInOp("b")->set_op_name(matmul_b_op.name());
    DiffLbi4BnInOp("b")->set_blob_name("out");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kMatmulConf, &GenerateBackwardOpConf);

}  // namespace oneflow
