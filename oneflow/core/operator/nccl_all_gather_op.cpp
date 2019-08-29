#include "oneflow/core/operator/nccl_all_gather_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

void NcclAllGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_all_gather_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& NcclAllGatherOp::GetCustomizedConf() const {
  return op_conf().nccl_all_gather_conf();
}

Maybe<void> NcclAllGatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* in_blob = GetBlobDesc4BnInOp(SoleIbn());
  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *in_blob;
  int64_t elem_cnt = in_blob->shape().elem_cnt();
  int64_t rank_num = parallel_ctx->rank_ctx().rank_num();
  out_blob->mut_shape() = Shape({elem_cnt * rank_num});
  return Maybe<void>::Ok();
}

LogicalBlobId NcclAllGatherOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name("out");
  return ret;
}

REGISTER_OP(OperatorConf::kNcclAllGatherConf, NcclAllGatherOp);

}  // namespace oneflow
