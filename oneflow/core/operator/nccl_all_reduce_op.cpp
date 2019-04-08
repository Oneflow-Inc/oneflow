#include "oneflow/core/operator/nccl_all_reduce_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclAllReduceOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_all_reduce_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& NcclAllReduceOp::GetCustomizedConf() const {
  return op_conf().nccl_all_reduce_conf();
}

void NcclAllReduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* in_blob = GetBlobDesc4BnInOp(SoleIbn());
  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *in_blob;
}

LogicalBlobId NcclAllReduceOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name("out");
  return ret;
}

LogicalNode* NcclAllReduceOp::NewProperLogicalNode() { return new NcclAllReduceLogicalNode(); }

REGISTER_OP(OperatorConf::kNcclAllReduceConf, NcclAllReduceOp);

}  // namespace oneflow
