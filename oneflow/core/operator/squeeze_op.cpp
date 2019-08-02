#include "oneflow/core/operator/squeeze_op.h"

namespace oneflow {

void SqueezeOp::InitFromOpConf() {
  CHECK(op_conf().has_squeeze_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& SqueezeOp::GetCustomizedConf() const { return op_conf().squeeze_conf(); }

void SqueezeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK(!in->has_instance_shape_field());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  auto dim_vec = in->shape().dim_vec();
  for (const auto& idx : PbRf2StdVec(op_conf().squeeze_conf().axis())) {
    // do not allow squeeze the first axis for now
    CHECK_GT(idx, 0);
    CHECK_LT(idx, dim_vec.size());
    CHECK_EQ(dim_vec[idx], 1);
    dim_vec[idx] = -1;
  }
  std::remove(dim_vec.begin(), dim_vec.end(), -1);
  out->mut_shape() = Shape(dim_vec);
}

REGISTER_OP(OperatorConf::kSqueezeConf, SqueezeOp);

}  // namespace oneflow
