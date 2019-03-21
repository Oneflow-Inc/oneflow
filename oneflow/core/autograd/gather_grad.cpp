#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_gather_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    const BlobDesc& in_logical_blob_desc = LogicalBlobDesc4BnInOp("in");
    const int64_t axis =
        op.op_conf().gather_conf().axis() < 0
            ? in_logical_blob_desc.shape().NumAxes() + op.op_conf().gather_conf().axis()
            : op.op_conf().gather_conf().axis();
    // TODO: support all axis
    CHECK_EQ(axis, 0);
    const int64_t gather_dim_size = in_logical_blob_desc.shape().At(axis);
    OperatorConf gather_grad_op;
    gather_grad_op.set_name(op.op_name() + "_grad");
    GatherGradOpConf* conf = gather_grad_op.mutable_gather_grad_conf();
    conf->set_axis(axis);
    conf->set_gather_dim_size(gather_dim_size);
    conf->set_indices(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_out_diff(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    conf->set_in_diff("in_diff");
    op_confs->push_back(gather_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(gather_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("in_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kGatherConf, &GenerateBackwardOpConf);

}  // namespace oneflow
