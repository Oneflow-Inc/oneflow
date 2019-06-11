#include "oneflow/core/operator/logical_and_op.h"

namespace oneflow {

void LogicalAndOp::InitFromOpConf() {
  CHECK(op_conf().has_logical_and_conf());
  EnrollInputBn("lhs", false);
  EnrollInputBn("rhs", false);
  EnrollOutputBn("out", false);
}

const PbMessage& LogicalAndOp::GetCustomizedConf() const {
  return this->op_conf().logical_and_conf();
}

void LogicalAndOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  // input: lhs and rhs
  const BlobDesc* lhs = GetBlobDesc4BnInOp("lhs");
  const BlobDesc* rhs = GetBlobDesc4BnInOp("rhs");
  CHECK_EQ(lhs->shape(), rhs->shape());
  CHECK_EQ(lhs->has_dim0_valid_num_field(), rhs->has_dim0_valid_num_field());
  CHECK_EQ(lhs->has_instance_shape_field(), rhs->has_instance_shape_field());
  // output
  *GetBlobDesc4BnInOp("out") = *lhs;
}

REGISTER_OP(OperatorConf::kLogicalAndConf, LogicalAndOp);

}  // namespace oneflow
