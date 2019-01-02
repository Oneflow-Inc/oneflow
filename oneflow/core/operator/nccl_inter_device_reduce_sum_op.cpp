#include "oneflow/core/operator/nccl_inter_device_reduce_sum_op.h"

namespace oneflow {

namespace {

int64_t GetBufferElemCnt(const BlobDesc* in, const ParallelContext* parallel_ctx) {
  return RoundUp(static_cast<size_t>(in->shape().elem_cnt()),
                 static_cast<size_t>(parallel_ctx->parallel_num()));
}

}  // namespace

void NcclInterDeviceReduceSumOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_inter_device_reduce_sum_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollFwBufBn("fw_buf");
  EnrollBwBufBn("bw_buf");
}

const PbMessage& NcclInterDeviceReduceSumOp::GetCustomizedConf() const {
  return op_conf().nccl_inter_device_reduce_sum_conf();
}

void NcclInterDeviceReduceSumOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in;
  const int64_t buffer_elem_cnt = GetBufferElemCnt(in, parallel_ctx);
  BlobDesc* fw_buf = GetBlobDesc4BnInOp("fw_buf");
  *fw_buf = *in;
  fw_buf->mut_shape() = Shape({buffer_elem_cnt});
}

void NcclInterDeviceReduceSumOp::InferBwBufBlobDescs(
    std::function<oneflow::BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const oneflow::ParallelContext* parallel_ctx, const oneflow::OpContext*) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const int64_t buffer_elem_cnt = GetBufferElemCnt(in, parallel_ctx);
  BlobDesc* bw_buf = GetBlobDesc4BnInOp("bw_buf");
  *bw_buf = *in;
  bw_buf->mut_shape() = Shape({buffer_elem_cnt});
}

}  // namespace oneflow
