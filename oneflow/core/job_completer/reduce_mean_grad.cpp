#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

DimVector KeepDims(const DimVector& dim_vec, const AxisVector& axis_vec) {
  DimVector ret = dim_vec;
  for (const auto& axis : axis_vec) { ret[axis] = 1; }
  return ret;
}

AxisVector ShiftAxisIfNegative(AxisVector axis_vec, const int64_t num_axes) {
  FOR_RANGE(size_t, i, 0, axis_vec.size()) {
    if (axis_vec[i] < 0) { axis_vec[i] += num_axes; }
    CHECK_LT(axis_vec[i], num_axes);
    CHECK_GE(axis_vec[i], 0);
  }
  return axis_vec;
}

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_reduce_mean_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf reduce_mean_grad_op;
    reduce_mean_grad_op.set_name(op.op_name() + "_grad");
    ReduceMeanGradOpConf* reduce_mean_grad_op_conf =
        reduce_mean_grad_op.mutable_reduce_mean_grad_conf();
    reduce_mean_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    reduce_mean_grad_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    reduce_mean_grad_op_conf->set_dx("dx");
    const ReduceMeanOpConf& reduce_mean_op_conf = op.op_conf().reduce_mean_conf();
    reduce_mean_grad_op_conf->mutable_reduced_axis()->CopyFrom(reduce_mean_op_conf.axis());
    op_confs->push_back(reduce_mean_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(reduce_mean_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kReduceMeanConf, &GenerateBackwardOpConf);

}  // namespace oneflow
