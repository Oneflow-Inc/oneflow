#include "oneflow/core/operator/assign_op.h"

namespace oneflow {

void AssignOp::InitFromOpConf() {
  CHECK(op_conf().has_assign_conf());
  EnrollInputBn("x");
  EnrollInputBn("y", false)->set_is_mutable(true);
}

void AssignOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  CHECK(*GetBlobDesc4BnInOp("x") == *GetBlobDesc4BnInOp("y"));
}

const PbMessage& AssignOp::GetCustomizedConf() const { return op_conf().assign_conf(); }

REGISTER_OP(OperatorConf::kAssignConf, AssignOp);

}  // namespace oneflow
