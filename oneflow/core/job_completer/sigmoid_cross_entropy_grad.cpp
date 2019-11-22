#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sigmoid_cross_entropy_loss_conf());
  if (DiffLbi4BnInOp("prediction") != nullptr) {
    OperatorConf sigmoid_cross_entropy_loss_grad_op;
    sigmoid_cross_entropy_loss_grad_op.set_name(op.op_name() + "_grad");

    SigmoidCrossEntropyLossGradOpConf* sigmoid_cross_entropy_loss_grad_op_conf =
        sigmoid_cross_entropy_loss_grad_op.mutable_sigmoid_cross_entropy_loss_grad_conf();
    sigmoid_cross_entropy_loss_grad_op_conf->set_label(GenLogicalBlobName(op.BnInOp2Lbi("label")));
    sigmoid_cross_entropy_loss_grad_op_conf->set_prediction(
        GenLogicalBlobName(op.BnInOp2Lbi("prediction")));
    sigmoid_cross_entropy_loss_grad_op_conf->set_loss_diff(
        GenLogicalBlobName(*DiffLbi4BnInOp("loss")));
    sigmoid_cross_entropy_loss_grad_op_conf->set_prediction_diff("prediction_diff");

    op_confs->push_back(sigmoid_cross_entropy_loss_grad_op);
    DiffLbi4BnInOp("prediction")->set_op_name(sigmoid_cross_entropy_loss_grad_op.name());
    DiffLbi4BnInOp("prediction")->set_blob_name("prediction_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSigmoidCrossEntropyLossConf, &GenerateBackwardOpConf);

}  // namespace oneflow
