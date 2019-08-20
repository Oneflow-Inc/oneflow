#include "oneflow/core/operator/expand_dims_op.h"

namespace oneflow {

void ExpandDimsOp::InitFromOpConf() {
  CHECK(op_conf().has_expand_dims_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ExpandDimsOp::GetCustomizedConf() const { return op_conf().expand_dims_conf(); }

void ExpandDimsOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  const int32_t axis = op_conf().expand_dims_conf().axis();
  std::vector<int64_t> dim_vec = in->shape().dim_vec();
  // do not allow expand the first dim
  CHECK_GT(axis, 0);
  CHECK_LE(axis, dim_vec.size());
  dim_vec.insert(dim_vec.begin() + axis, 1);
  out->mut_shape() = Shape(dim_vec);
}

REGISTER_OP(OperatorConf::kExpandDimsConf, ExpandDimsOp);

}  // namespace oneflow
