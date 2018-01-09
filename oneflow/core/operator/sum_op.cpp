#include "oneflow/core/operator/sum_op.h"

namespace oneflow {

void SumOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& SumOp::GetSpecialConf() const { return op_conf().sum_conf(); }

void SumOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext*) const {
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  std::vector<int64_t> out_dim_vec = in_blob->shape().dim_vec();
  int32_t axis = GetCorrectAxis(GetBlobDesc4BnInOp);
  out_dim_vec.erase(out_dim_vec.begin() + axis);
  if (out_dim_vec.empty()) { out_dim_vec.push_back(1); }
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob->mut_shape() = Shape(out_dim_vec);
  out_blob->set_data_type(in_blob->data_type());
  out_blob->set_has_data_id_field(in_blob->has_data_id_field() && axis > 0);
}

void SumOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext*, KernelConf* kernel_conf) const {
  kernel_conf->mutable_sum_conf()->set_axis(GetCorrectAxis(GetBlobDesc4BnInOp));
}

int32_t SumOp::GetCorrectAxis(std::function<const BlobDesc*(const std::string&)>
                                  GetBlobDesc4BnInOp) const {
  int32_t axis = op_conf().sum_conf().axis();
  if (axis < 0) { axis += GetBlobDesc4BnInOp("in")->shape().NumAxes(); }
  return axis;
}

REGISTER_OP(OperatorConf::kSumConf, SumOp);

}  // namespace oneflow
