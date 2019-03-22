#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_batch_gather_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf batch_gather_grad;
    batch_gather_grad.set_name(op.op_name() + "_grad");
    BatchGatherGradOpConf* conf = batch_gather_grad.mutable_batch_gather_grad_conf();
    conf->set_out_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_indices(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_in_diff("in_diff");
    const Shape& in_shape = LogicalBlobDesc4BnInOp("in").shape();
    const Shape& indices_shape = LogicalBlobDesc4BnInOp("indices").shape();
    CHECK_GE(in_shape.NumAxes(), indices_shape.NumAxes());
    CHECK_GE(indices_shape.NumAxes(), 1);
    conf->set_gather_dim_size(in_shape.At(indices_shape.NumAxes() - 1));
    op_confs->push_back(batch_gather_grad);
    DiffLbi4BnInOp("in")->set_op_name(batch_gather_grad.name());
    DiffLbi4BnInOp("in")->set_blob_name("in_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBatchGatherConf, &GenerateBackwardOpConf);

}  // namespace oneflow
