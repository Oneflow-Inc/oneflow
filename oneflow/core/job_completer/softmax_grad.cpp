#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

Maybe<void> GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_softmax_conf());
  const int32_t axis = op.op_conf().softmax_conf().axis();
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf softmax_grad_op;
    softmax_grad_op.set_name(op.op_name() + "_grad");
    SoftmaxGradOpConf* softmax_grad_op_conf = softmax_grad_op.mutable_softmax_grad_conf();
    softmax_grad_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    softmax_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    if (axis != -1 && axis != (LogicalBlobDesc4BnInOp("out").shape().NumAxes() - 1)) {
      softmax_grad_op_conf->set_transpose_x(GenLogicalBlobName(op.BnInOp2Lbi("transpose_in")));
      softmax_grad_op_conf->set_transpose_y(GenLogicalBlobName(op.BnInOp2Lbi("transpose_out")));
    }
    softmax_grad_op_conf->set_dx("dx");
    softmax_grad_op_conf->set_axis(axis);
    op_confs->push_back(softmax_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(softmax_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSoftmaxConf, &GenerateBackwardOpConf);

}  // namespace oneflow
