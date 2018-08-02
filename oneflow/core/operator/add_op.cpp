#include "oneflow/core/operator/add_op.h"

namespace oneflow {

void AddOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_add_conf());
  EnrollBwBufBn(k_bw_activation_blob_name);
}
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }
void AddOp::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext*, const OpContext* op_ctx) const {
  *GetBlobDesc4BnInOp(k_bw_activation_blob_name) = *GetBlobDesc4BnInOp("out");
}

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
