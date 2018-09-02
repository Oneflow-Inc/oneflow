#include "oneflow/core/operator/nccl_allreduce_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

void NcclAllreduceOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_allreduce_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& NcclAllreduceOp::GetCustomizedConf() const {
  return op_conf().nccl_allreduce_conf();
}

void NcclAllreduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* in_blob = GetBlobDesc4BnInOp(SoleIbn());
  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *in_blob;
}

LogicalBlobId NcclAllreduceOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name("out");
  return ret;
}

REGISTER_OP(OperatorConf::kNcclAllreduceConf, NcclAllreduceOp);

}  // namespace oneflow
