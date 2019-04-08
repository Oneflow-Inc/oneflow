#include "oneflow/core/operator/where_op.h"

namespace oneflow {

void WhereOp::InitFromOpConf() {
  CHECK(op_conf().has_where_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("x");
  EnrollInputBn("y");
  EnrollOutputBn("out");
}

const PbMessage& WhereOp::GetCustomizedConf() const { return op_conf().where_conf(); }

void WhereOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("x");
}

REGISTER_OP(OperatorConf::kWhereConf, WhereOp);

}  // namespace oneflow
