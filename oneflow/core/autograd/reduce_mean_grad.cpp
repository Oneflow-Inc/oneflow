#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

std::vector<int64_t> KeepDims(const std::vector<int64_t> dim_vec,
                              const std::vector<int64_t> axis_vec) {
  std::vector<int64_t> ret = dim_vec;
  for (const auto& axis : axis_vec) { ret[axis] = 1; }
  return ret;
}

std::vector<int64_t> ShiftAxisIfNegative(std::vector<int64_t> axis_vec, const int64_t num_axes) {
  FOR_RANGE(size_t, i, 0, axis_vec.size()) {
    if (axis_vec[i] < 0) { axis_vec[i] += num_axes; }
    CHECK_LT(axis_vec[i], num_axes);
    CHECK_GE(axis_vec[i], 0);
  }
  return axis_vec;
}

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_reduce_sum_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf reduce_mean_grad_op;
    reduce_mean_grad_op.set_name(op.op_name() + "_grad");
    ReduceMeanGradOpConf* reduce_mean_grad_op_conf = reduce_mean_grad_op.mutable_reduce_mean_grad_conf();
    reduce_mean_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    reduce_mean_grad_op_conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    reduce_mean_grad_op_conf->set_dx("dx");
    if (op.op_conf().reduce_mean_conf().axis().empty() == false
        && op.op_conf().reduce_mean_conf().keep_dims() == false) {
      std::vector<int64_t> kept_dims;
      const PbRf<int32_t>& axis_repeated = op.op_conf().reduce_mean_conf().axis();
      std::vector<int64_t> axis_vec = {axis_repeated.begin(), axis_repeated.end()};
      const BlobDesc& in_blob = LogicalBlobDesc4BnInOp("in");
      kept_dims = KeepDims(in_blob.shape().dim_vec(),
                           ShiftAxisIfNegative(axis_vec, in_blob.shape().NumAxes()));
      Shape(kept_dims).ToProto(reduce_mean_grad_op_conf->mutable_kept_dims_shape());
    }
    op_confs->push_back(reduce_mean_grad_op);
    DiffLbi4BnInOp("in")->set_op_name(reduce_mean_grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("dx");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kReduceMeanConf, &GenerateBackwardOpConf);

}  // namespace oneflow
