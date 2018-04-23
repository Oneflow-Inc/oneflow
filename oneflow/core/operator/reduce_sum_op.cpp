#include "oneflow/core/operator/reduce_sum_op.h"

namespace oneflow {

void ReduceSumOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
  // For Parallel
  if (op_conf().reduce_sum_conf().has_axis() == false) { EnrollDataTmpBn("tmp"); }
}

const PbMessage& ReduceSumOp::GetCustomizedConf() const { return op_conf().reduce_sum_conf(); }

void ReduceSumOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  std::vector<int64_t> out_dim_vec = {1};
  if (op_conf().reduce_sum_conf().has_axis()) {
    out_dim_vec = in_blob->shape().dim_vec();
    int32_t axis = GetCorrectAxis(GetBlobDesc4BnInOp);
    if (op_conf().reduce_sum_conf().keepdims() == true) {
      out_dim_vec[axis] = 1;
    } else {
      out_dim_vec.erase(out_dim_vec.begin() + axis);
    }
    if (out_dim_vec.empty()) { out_dim_vec.push_back(1); }
  } else {
    BlobDesc* tmp_blob = GetBlobDesc4BnInOp("tmp");
    tmp_blob->mut_shape() = Shape(in_blob->shape().dim_vec());
    tmp_blob->set_data_type(in_blob->data_type());
    tmp_blob->set_has_data_id_field(false);
  }
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  *out_blob = *in_blob;
  out_blob->mut_shape() = Shape(out_dim_vec);
  out_blob->set_has_data_id_field(in_blob->has_data_id_field()
                                  && op_conf().reduce_sum_conf().has_axis()
                                  && GetCorrectAxis(GetBlobDesc4BnInOp) > 0);
}

void ReduceSumOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  if (op_conf().reduce_sum_conf().has_axis() == false) { return; }
  kernel_conf->mutable_reduce_sum_conf()->set_axis(GetCorrectAxis(GetBlobDesc4BnInOp));
}

int32_t ReduceSumOp::GetCorrectAxis(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const {
  int32_t axis = op_conf().reduce_sum_conf().axis();
  if (axis < 0) { axis += GetBlobDesc4BnInOp("in")->shape().NumAxes(); }
  return axis;
}

REGISTER_OP(OperatorConf::kReduceSumConf, ReduceSumOp);

}  // namespace oneflow
