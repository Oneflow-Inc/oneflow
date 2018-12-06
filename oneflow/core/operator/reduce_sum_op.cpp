#include "oneflow/core/operator/reduce_sum_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace {
std::vector<int64_t> KeepDims(const std::vector<int64_t> dim_vec,
                              const std::vector<int64_t> axis_vec) {
  std::vector<int64_t> ret = dim_vec;
  for (const auto& axis : axis_vec) { ret[axis] = 1; }
  return ret;
}

std::vector<int64_t> DropDims(const std::vector<int64_t> dim_vec,
                              const std::vector<int64_t> axis_vec) {
  std::vector<int64_t> ret;
  std::vector<int32_t> dim2is_reduced(dim_vec.size());
  for (const auto& axis : axis_vec) { dim2is_reduced[axis] = 1; }
  FOR_RANGE(int64_t, i, 0, dim_vec.size()) {
    if (dim2is_reduced[i] != 1) { ret.push_back(dim_vec[i]); }
  }
  if (ret.empty()) { ret.push_back(1); }
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

}  // namespace

void ReduceSumOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (op_conf().reduce_sum_conf().has_in_sys()) {
    EnrollDataTmpBn("fw_tmp");
  } else {
    EnrollFwBufBn("fw_tmp");
  }
}

const PbMessage& ReduceSumOp::GetCustomizedConf() const { return op_conf().reduce_sum_conf(); }

void ReduceSumOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
  std::vector<int64_t> out_dim_vec;
  if (conf.axis().empty()) {
    if (conf.keep_dims() == true) {
      out_dim_vec.resize(in_blob->shape().NumAxes());
      std::fill(out_dim_vec.begin(), out_dim_vec.end(), 1);
    } else {
      out_dim_vec = {1};
    }
  } else {
    const PbRf<int32_t>& axis_repeated = conf.axis();
    std::vector<int64_t> axis_vec = {axis_repeated.begin(), axis_repeated.end()};
    axis_vec = ShiftAxisIfNegative(axis_vec, in_blob->shape().NumAxes());
    std::sort(axis_vec.begin(), axis_vec.end());
    CHECK(std::unique(axis_vec.begin(), axis_vec.end()) == axis_vec.end())
        << "duplicate found in axis";
    if (conf.keep_dims() == true) {
      out_dim_vec = KeepDims(in_blob->shape().dim_vec(), axis_vec);
    } else {
      out_dim_vec = DropDims(in_blob->shape().dim_vec(), axis_vec);
    }
  }
  CHECK(!out_dim_vec.empty());
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob->set_data_type(in_blob->data_type());
  out_blob->mut_shape() = Shape(out_dim_vec);
}

void ReduceSumOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  std::vector<int64_t> kept_dims;
  if (conf.axis().empty()) {
    kept_dims.resize(in_blob->shape().NumAxes());
    std::fill(kept_dims.begin(), kept_dims.end(), 1);
  } else {
    const PbRf<int32_t>& axis_repeated = op_conf().reduce_sum_conf().axis();
    std::vector<int64_t> axis_vec = {axis_repeated.begin(), axis_repeated.end()};
    kept_dims = KeepDims(in_blob->shape().dim_vec(),
                         ShiftAxisIfNegative(axis_vec, in_blob->shape().NumAxes()));
  }
  Shape(kept_dims).ToProto(kernel_conf->mutable_reduce_sum_conf()->mutable_kept_dims_shape());
}

REGISTER_OP(OperatorConf::kReduceSumConf, ReduceSumOp);

}  // namespace oneflow
