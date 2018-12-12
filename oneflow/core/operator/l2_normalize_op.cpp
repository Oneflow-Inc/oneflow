#include "oneflow/core/operator/l2_normalize_op.h"

namespace oneflow {

void L2NormalizeOp::InitFromOpConf() {
  CHECK(op_conf().has_l2_normalize_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& L2NormalizeOp::GetCustomizedConf() const { return op_conf().l2_normalize_conf(); }

void L2NormalizeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const L2NormalizeOpConf& conf = op_conf().l2_normalize_conf();
  CHECK_GT(conf.axis(), 0);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
}

REGISTER_OP(OperatorConf::kL2NormalizeConf, L2NormalizeOp);

}  // namespace oneflow
