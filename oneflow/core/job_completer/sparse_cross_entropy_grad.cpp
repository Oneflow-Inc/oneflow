#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sparse_cross_entropy_conf());
  if (DiffLbi4BnInOp("prediction") != nullptr) {
    OperatorConf sparse_cross_entropy_grad_op;
    sparse_cross_entropy_grad_op.set_name(op.op_name() + "_grad");
    SparseCrossEntropyGradOpConf* sparse_cross_entropy_grad_op_conf =
        sparse_cross_entropy_grad_op.mutable_sparse_cross_entropy_grad_conf();
    sparse_cross_entropy_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    sparse_cross_entropy_grad_op_conf->set_label(GenLogicalBlobName(op.BnInOp2Lbi("label")));
    sparse_cross_entropy_grad_op_conf->set_prediction(
        GenLogicalBlobName(op.BnInOp2Lbi("prediction")));
    sparse_cross_entropy_grad_op_conf->set_prediction_diff("prediction_diff");
    op_confs->push_back(sparse_cross_entropy_grad_op);
    DiffLbi4BnInOp("prediction")->set_op_name(sparse_cross_entropy_grad_op.name());
    DiffLbi4BnInOp("prediction")->set_blob_name("prediction_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSparseCrossEntropyConf, &GenerateBackwardOpConf);

}  // namespace oneflow
