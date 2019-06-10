#include "oneflow/core/operator/where_op.h"

namespace oneflow {

void WhereOp::InitFromOpConf() {
  CHECK(op_conf().has_where_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("lhs");
  EnrollInputBn("rhs");
  EnrollOutputBn("out");
}

const PbMessage& WhereOp::GetCustomizedConf() const { return this->op_conf().where_conf(); }

void WhereOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  WhereKernelConf* conf = kernel_conf->mutable_where_conf();
  conf->set_cond_type(GetBlobDesc4BnInOp("condition")->data_type());
  conf->set_value_type(GetBlobDesc4BnInOp("lhs")->data_type());
}

void WhereOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  // input: condition
  const BlobDesc* condition = GetBlobDesc4BnInOp("condition");
  CHECK(IsIntegralDataType(condition->data_type()));
  // input: lhs and rhs
  const BlobDesc* lhs = GetBlobDesc4BnInOp("lhs");
  const BlobDesc* rhs = GetBlobDesc4BnInOp("rhs");
  const Shape shape = condition->shape();
  const bool dim0_varing = condition->has_dim0_valid_num_field();
  const bool has_instance_shape = condition->has_instance_shape_field();
  CHECK_EQ(shape, lhs->shape());
  CHECK_EQ(shape, rhs->shape());
  CHECK_EQ(dim0_varing, lhs->has_dim0_valid_num_field());
  CHECK_EQ(dim0_varing, rhs->has_dim0_valid_num_field());
  CHECK_EQ(has_instance_shape, lhs->has_instance_shape_field());
  CHECK_EQ(has_instance_shape, rhs->has_instance_shape_field());
  CHECK_EQ(lhs->data_type(), rhs->data_type());
  // output
  *GetBlobDesc4BnInOp("out") = *lhs;
}

REGISTER_OP(OperatorConf::kWhereConf, WhereOp);

}  // namespace oneflow
