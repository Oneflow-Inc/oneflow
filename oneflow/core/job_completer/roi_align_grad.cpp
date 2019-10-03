#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_roi_align_conf());
  if (DiffLbi4BnInOp("x") != nullptr) {
    OperatorConf roi_align_grad_op;
    roi_align_grad_op.set_name(op.op_name() + "_grad");
    RoiAlignGradOpConf* roi_align_grad_op_conf = roi_align_grad_op.mutable_roi_align_grad_conf();
    roi_align_grad_op_conf->set_x_like(GenLogicalBlobName(op.BnInOp2Lbi("x")));
    roi_align_grad_op_conf->set_rois(GenLogicalBlobName(op.BnInOp2Lbi("rois")));
    roi_align_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("y")));
    roi_align_grad_op_conf->set_dx("dx");
    *roi_align_grad_op_conf->mutable_roi_align_args() =
        op.op_conf().roi_align_conf().roi_align_args();
    op_confs->push_back(roi_align_grad_op);
    DiffLbi4BnInOp("x")->set_op_name(roi_align_grad_op.name());
    DiffLbi4BnInOp("x")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kRoiAlignConf, &GenerateBackwardOpConf);

}  // namespace oneflow
