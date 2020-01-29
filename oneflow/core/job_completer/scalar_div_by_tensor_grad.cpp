#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_scalar_div_by_tensor_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf scalar_div_by_tensor_grad_op;
    scalar_div_by_tensor_grad_op.set_name(op.op_name() + "_grad");
    ScalarDivByTensorOpConf* conf =
        scalar_div_by_tensor_grad_op.mutable_scalar_div_by_tensor_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_scalar(GenLogicalBlobName(op.BnInOp2Lbi("scalar")));
    conf->set_out("out");
    op_confs->push_back(scalar_div_by_tensor_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(scalar_div_by_tensor_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
  if (DiffLbi4BnInOp("scalar") != nullptr) {
    OperatorConf broadcast_div_grad_op;
    broadcast_div_grad_op.set_name(op.op_name() + "_grad_broadcast_div_grad");
    BroadcastDivGradOpConf* broadcast_div_grad_op_conf =
        broadcast_div_grad_op.mutable_broadcast_div_grad_conf();
    broadcast_div_grad_op_conf->set_b(GenLogicalBlobName(op.BnInOp2Lbi("scalar")));
    broadcast_div_grad_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    broadcast_div_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    broadcast_div_grad_op_conf->set_db("db");
    op_confs->push_back(broadcast_div_grad_op);
    DiffLbi4BnInOp("scalar")->set_op_name(broadcast_div_grad_op.name());
    DiffLbi4BnInOp("scalar")->set_blob_name(broadcast_div_grad_op_conf->db());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarDivByTensorConf, &GenerateBackwardOpConf);

}  // namespace oneflow
