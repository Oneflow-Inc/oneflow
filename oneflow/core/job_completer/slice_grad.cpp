#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_slice_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf slice_grad_op;
    slice_grad_op.set_name(op.op_name() + "_grad");
    SliceGradOpConf* slice_grad_op_conf = slice_grad_op.mutable_slice_grad_conf();
    slice_grad_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    slice_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    slice_grad_op_conf->set_dx("dx");
    slice_grad_op_conf->mutable_dim_slice_conf()->CopyFrom(
        op.op_conf().slice_conf().dim_slice_conf());
    op_confs->push_back(slice_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(slice_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSliceConf, &GenerateBackwardOpConf);

}  // namespace oneflow
