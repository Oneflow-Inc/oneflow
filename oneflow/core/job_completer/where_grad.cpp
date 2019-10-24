#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_where_conf());
  if (DiffLbi4BnInOp("x") != nullptr) {
    OperatorConf where_x_grad_op;
    where_x_grad_op.set_name(op.op_name() + "_x_grad");
    WhereXGradOpConf* where_x_grad_op_conf = where_x_grad_op.mutable_where_x_grad_conf();
    where_x_grad_op_conf->set_condition(GenLogicalBlobName(op.BnInOp2Lbi("condition")));
    where_x_grad_op_conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    where_x_grad_op_conf->set_x_diff("x_diff");
    op_confs->push_back(where_x_grad_op);
    DiffLbi4BnInOp("x")->set_op_name(where_x_grad_op.name());
    DiffLbi4BnInOp("x")->set_blob_name("x_diff");
  }
  if (DiffLbi4BnInOp("y") != nullptr) {
    OperatorConf where_y_grad_op;
    where_y_grad_op.set_name(op.op_name() + "_y_grad");
    WhereYGradOpConf* where_y_grad_op_conf = where_y_grad_op.mutable_where_y_grad_conf();
    where_y_grad_op_conf->set_condition(GenLogicalBlobName(op.BnInOp2Lbi("condition")));
    where_y_grad_op_conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    where_y_grad_op_conf->set_y_diff("y_diff");
    op_confs->push_back(where_y_grad_op);
    DiffLbi4BnInOp("y")->set_op_name(where_y_grad_op.name());
    DiffLbi4BnInOp("y")->set_blob_name("y_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kWhereConf, &GenerateBackwardOpConf);

}  // namespace oneflow
