#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_where_conf());
  OperatorConf zeros_like_op;
  zeros_like_op.set_name(op.op_name() + "_zeros");
  ZerosLikeOpConf* zeros_like_op_conf = zeros_like_op.mutable_zeros_like_conf();
  zeros_like_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("x")));
  zeros_like_op_conf->set_out("out");
  op_confs->push_back(zeros_like_op);

  if (DiffLbi4BnInOp("x") != nullptr) {
    OperatorConf where_x_grad_op;
    where_x_grad_op.set_name(op.op_name() + "_x_grad");
    WhereOpConf* where_x_grad_op_conf = where_x_grad_op.mutable_where_conf();
    where_x_grad_op_conf->set_condition(GenLogicalBlobName(op.BnInOp2Lbi("condition")));
    where_x_grad_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    where_x_grad_op_conf->set_y(zeros_like_op.name() + "/out");
    where_x_grad_op_conf->set_out("out");
    op_confs->push_back(where_x_grad_op);

    DiffLbi4BnInOp("x")->set_op_name(where_x_grad_op.name());
    DiffLbi4BnInOp("x")->set_blob_name(where_x_grad_op_conf->out());
  }
  if (DiffLbi4BnInOp("y") != nullptr) {
    OperatorConf where_y_grad_op;
    where_y_grad_op.set_name(op.op_name() + "_y_grad");
    WhereOpConf* where_y_grad_op_conf = where_y_grad_op.mutable_where_conf();
    where_y_grad_op_conf->set_condition(GenLogicalBlobName(op.BnInOp2Lbi("condition")));
    where_y_grad_op_conf->set_x(zeros_like_op.name() + "/out");
    where_y_grad_op_conf->set_y(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    where_y_grad_op_conf->set_out("out");
    op_confs->push_back(where_y_grad_op);

    DiffLbi4BnInOp("y")->set_op_name(where_y_grad_op.name());
    DiffLbi4BnInOp("y")->set_blob_name(where_y_grad_op_conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kWhereConf, &GenerateBackwardOpConf);

}  // namespace oneflow
