#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(const Operator& op, std::vector<OperatorConf>* op_confs,
                            const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
                            const std::function<DataType(const std::string&)>& DataType4BnInOp) {
  CHECK(op.op_conf().has_softmax_conf());
  if (DiffLbi4BnInOp("x") != nullptr) {
    OperatorConf softmax_grad_op;
    softmax_grad_op.set_name(op.op_name() + "_grad");
    SoftmaxGradOpConf* softmax_grad_op_conf = softmax_grad_op.mutable_softmax_grad_conf();
    softmax_grad_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("y")));
    softmax_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("y")));
    softmax_grad_op_conf->set_transpose_x(GenLogicalBlobName(op.BnInOp2Lbi("transpose_x")));
    softmax_grad_op_conf->set_transpose_y(GenLogicalBlobName(op.BnInOp2Lbi("transpose_y")));
    softmax_grad_op_conf->set_dx("dx");
    softmax_grad_op_conf->set_axis(op.op_conf().softmax_conf().axis());
    op_confs->push_back(softmax_grad_op);
    DiffLbi4BnInOp("x")->set_op_name(softmax_grad_op.name());
    DiffLbi4BnInOp("x")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSoftmaxConf, &GenerateBackwardOpConf);

}  // namespace oneflow
