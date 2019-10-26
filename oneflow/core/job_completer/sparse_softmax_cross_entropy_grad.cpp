#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sparse_softmax_cross_entropy_conf());
  if (DiffLbi4BnInOp("prediction") != nullptr) {
    OperatorConf sparse_softmax_cross_entropy_grad_op;
    sparse_softmax_cross_entropy_grad_op.set_name(op.op_name() + "_grad");
    SparseSoftmaxCrossEntropyGradOpConf* sparse_softmax_cross_entropy_grad_op_conf =
        sparse_softmax_cross_entropy_grad_op.mutable_sparse_softmax_cross_entropy_grad_conf();
    sparse_softmax_cross_entropy_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    sparse_softmax_cross_entropy_grad_op_conf->set_label(
        GenLogicalBlobName(op.BnInOp2Lbi("label")));
    sparse_softmax_cross_entropy_grad_op_conf->set_prob(GenLogicalBlobName(op.BnInOp2Lbi("prob")));
    sparse_softmax_cross_entropy_grad_op_conf->set_dx("dx");
    op_confs->push_back(sparse_softmax_cross_entropy_grad_op);
    DiffLbi4BnInOp("prediction")->set_op_name(sparse_softmax_cross_entropy_grad_op.name());
    DiffLbi4BnInOp("prediction")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSparseSoftmaxCrossEntropyConf, &GenerateBackwardOpConf);

}  // namespace oneflow
