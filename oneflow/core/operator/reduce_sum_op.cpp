#include "oneflow/core/operator/reduce_sum_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void ReduceSumOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("fw_tmp");
}

const PbMessage& ReduceSumOp::GetCustomizedConf() const { return op_conf().reduce_sum_conf(); }

void ReduceSumOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
  std::vector<int64_t> out_dim_vec;
  if (conf.axis_size() == 0) {
    if (conf.keepdims() == true) {
      out_dim_vec.resize(in_blob->shape().NumAxes());
      std::fill(out_dim_vec.begin(), out_dim_vec.end(), 1);
    } else {
      out_dim_vec = {1};
    }
  } else {
    auto axis_repeated = conf.axis();
    std::vector<int64_t> axis_vec = {axis_repeated.begin(), axis_repeated.end()};
    FOR_RANGE(size_t, i, 0, axis_vec.size()) {
      axis_vec[i] = GetCorrectAxis(axis_vec[i], GetBlobDesc4BnInOp);
    }
    std::sort(axis_vec.begin(), axis_vec.end());
    CHECK(std::unique(axis_vec.begin(), axis_vec.end()) == axis_vec.end())
        << "duplicate found in axis";
    if (conf.keepdims() == true) {
      out_dim_vec = KeptDims(GetBlobDesc4BnInOp);
    } else {
      out_dim_vec = OutDims(GetBlobDesc4BnInOp);
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
  std::vector<int64_t> kept_dims = KeptDims(GetBlobDesc4BnInOp);
  *kernel_conf->mutable_reduce_sum_conf()->mutable_kept_dims_shape()->mutable_dim() = {
      kept_dims.begin(), kept_dims.end()};
}

std::vector<int64_t> ReduceSumOp::KeptDims(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const {
  std::vector<int64_t> ret = GetBlobDesc4BnInOp("in")->shape().dim_vec();
  for (const auto& axis : op_conf().reduce_sum_conf().axis()) {
    ret[GetCorrectAxis(axis, GetBlobDesc4BnInOp)] = 1;
  }
  return ret;
}

std::vector<int64_t> ReduceSumOp::OutDims(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const {
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  std::vector<int64_t> ret;
  std::set<int64_t> correct_axis;
  for (const auto& axis : op_conf().reduce_sum_conf().axis()) {
    correct_axis.insert(GetCorrectAxis(axis, GetBlobDesc4BnInOp));
  }
  FOR_RANGE(int64_t, i, 0, in_blob->shape().NumAxes()) {
    if (std::find(correct_axis.begin(), correct_axis.end(), i) == correct_axis.end())
      ret.push_back(in_blob->shape().dim_vec()[i]);
  }
  if (ret.empty()) { ret.push_back(1); }
  return ret;
}

int64_t ReduceSumOp::GetCorrectAxis(
    int64_t axis, std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const {
  const int64_t num_axes = GetBlobDesc4BnInOp("in")->shape().NumAxes();
  if (axis < 0) { axis += num_axes; }
  CHECK_LT(axis, num_axes);
  CHECK_GE(axis, 0);
  return axis;
}

REGISTER_OP(OperatorConf::kReduceSumConf, ReduceSumOp);

}  // namespace oneflow
