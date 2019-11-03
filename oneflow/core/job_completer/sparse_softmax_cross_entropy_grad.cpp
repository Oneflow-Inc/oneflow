#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sparse_softmax_cross_entropy_conf());
  if (DiffLbi4BnInOp("prediction") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_grad");
    SparseSoftmaxCrossEntropyGradOpConf* conf =
        op_conf.mutable_sparse_softmax_cross_entropy_grad_conf();
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_label(GenLogicalBlobName(op.BnInOp2Lbi("label")));
    conf->set_prob(GenLogicalBlobName(op.BnInOp2Lbi("prob")));
    conf->set_dx("dx");
    op_confs->push_back(op_conf);
    DiffLbi4BnInOp("prediction")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("prediction")->set_blob_name(conf->dx());
  }
}

void GenerateMs1BackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sparse_softmax_cross_entropy_ms1_conf());
  if (DiffLbi4BnInOp("prediction") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_grad");
    SparseSoftmaxCrossEntropyMs1GradOpConf* conf =
        op_conf.mutable_sparse_softmax_cross_entropy_ms1_grad_conf();
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_label(GenLogicalBlobName(op.BnInOp2Lbi("label")));
    conf->set_prob(GenLogicalBlobName(op.BnInOp2Lbi("prob")));
    conf->set_dx("dx");
    conf->set_depth(op.op_conf().sparse_softmax_cross_entropy_ms1_conf().depth());
    op_confs->push_back(op_conf);
    DiffLbi4BnInOp("prediction")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("prediction")->set_blob_name(conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSparseSoftmaxCrossEntropyConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kSparseSoftmaxCrossEntropyMs1Conf, &GenerateMs1BackwardOpConf);

}  // namespace oneflow
