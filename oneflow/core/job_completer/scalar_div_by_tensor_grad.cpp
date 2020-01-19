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
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kScalarDivByTensorConf, &GenerateBackwardOpConf);

}  // namespace oneflow
