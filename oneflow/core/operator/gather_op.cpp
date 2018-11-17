#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("indices");
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetCustomizedConf() const { return op_conf().gather_conf(); }

void GatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GE(indices->shape().NumAxes(), 1);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GE(in->shape().NumAxes(), 1);
  int64_t axis = op_conf().gather_conf().axis();
  if (axis < 0) { axis += in->shape().NumAxes(); }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, in->shape().NumAxes());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  std::vector<int64_t> dim_vec;
  dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin(),
                 in->shape().dim_vec().cbegin() + axis);
  dim_vec.insert(dim_vec.end(), indices->shape().dim_vec().cbegin(),
                 indices->shape().dim_vec().cend());
  dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin() + axis + 1,
                 in->shape().dim_vec().end());
  out->mut_shape() = Shape(dim_vec);
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
