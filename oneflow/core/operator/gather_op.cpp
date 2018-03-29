#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetSpecialConf() const {
  return op_conf().gather_conf();
}

void GatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *GetBlobDesc4BnInOp("in");
  out->set_max_col_num(1);
  CHECK(out->has_col_num_field());
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
