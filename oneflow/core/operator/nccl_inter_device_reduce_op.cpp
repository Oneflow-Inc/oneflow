#include "oneflow/core/operator/nccl_inter_device_reduce_op.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

namespace {

int64_t GetRoundUpElemCnt(const BlobDesc* in, const ParallelContext* parallel_ctx) {
  return RoundUp(static_cast<size_t>(in->shape().elem_cnt()),
                 static_cast<size_t>(parallel_ctx->parallel_num()));
}

}  // namespace

void NcclInterDeviceReduceOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_inter_device_reduce_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollFwBufBn("fw_buf");
  EnrollBwBufBn("bw_buf");
}

const PbMessage& NcclInterDeviceReduceOp::GetCustomizedConf() const {
  return op_conf().nccl_inter_device_reduce_conf();
}

void NcclInterDeviceReduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in;
  const int64_t round_up_elem_cnt = GetRoundUpElemCnt(in, parallel_ctx);
  if (in->shape().elem_cnt() != round_up_elem_cnt) {
    BlobDesc* fw_buf = GetBlobDesc4BnInOp("fw_buf");
    *fw_buf = *in;
    fw_buf->mut_shape() = Shape({round_up_elem_cnt});
  }
}

void NcclInterDeviceReduceOp::InferBwBufBlobDescs(
    std::function<oneflow::BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const oneflow::ParallelContext* parallel_ctx, const oneflow::OpContext*) const {
  const BlobDesc* out_diff = GetBlobDesc4BnInOp(GenDiffBn("out"));
  const int64_t round_up_elem_cnt = GetRoundUpElemCnt(out_diff, parallel_ctx);
  if (out_diff->shape().elem_cnt() != round_up_elem_cnt) {
    BlobDesc* bw_buf = GetBlobDesc4BnInOp("bw_buf");
    *bw_buf = *out_diff;
    bw_buf->mut_shape() = Shape({round_up_elem_cnt});
  }
}

void NcclInterDeviceReduceOp::InferDiffBlobDescsWithoutFwBlob(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  BlobDesc* in_diff_blob_desc = GetBlobDesc4BnInOp(GenDiffBn("in"));
  BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp(GenDiffBn("out"));
  *in_diff_blob_desc = *out_diff_blob_desc;
}

LogicalNode* NcclInterDeviceReduceOp::NewProperLogicalNode() {
  return new NcclInterDeviceReduceForwardLogicalNode();
}

REGISTER_OP(OperatorConf::kNcclInterDeviceReduceConf, NcclInterDeviceReduceOp);

}  // namespace oneflow
