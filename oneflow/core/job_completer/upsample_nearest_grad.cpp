#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_upsample_nearest_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf upsample_nearest_grad_op;
    upsample_nearest_grad_op.set_name(op.op_name() + "_grad");
    UpsampleNearestGradOpConf* upsample_nearest_grad_op_conf =
        upsample_nearest_grad_op.mutable_upsample_nearest_grad_conf();
    upsample_nearest_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    upsample_nearest_grad_op_conf->set_dx("dx");
    upsample_nearest_grad_op_conf->set_scale(op.op_conf().upsample_nearest_conf().scale());
    op_confs->push_back(upsample_nearest_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(upsample_nearest_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kUpsampleNearestConf, &GenerateBackwardOpConf);

}  // namespace oneflow
