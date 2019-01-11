#include "oneflow/core/operator/boxing_align_broadcast_reshape_op.h"

namespace oneflow {

void BoxingAlignBroadcastReshapeOp::InitFromOpConf() {
  CHECK(op_conf().has_boxing_align_broadcast_reshape_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollBwBufBn("bw_buf");
  EnrollBwBufBn("reduce_buf");
}

const PbMessage& BoxingAlignBroadcastReshapeOp::GetCustomizedConf() const {
  return op_conf().boxing_align_broadcast_reshape_conf();
}

void BoxingAlignBroadcastReshapeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BoxingAlignBroadcastReshapeOpConf& conf = op_conf().boxing_align_broadcast_reshape_conf();
  CHECK_EQ(parallel_ctx->parallel_num(), 1);
  CHECK_GE(conf.target_parallel_num(), 1);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  const int64_t piece_size = Global<JobDesc>::Get()->PieceSize();
  CHECK_EQ(piece_size % conf.target_parallel_num(), 0);
  const int64_t target_piece_size = piece_size / conf.target_parallel_num();
  const int64_t aligned_elem_cnt = RoundUp(in->shape().elem_cnt(), target_piece_size);
  out->mut_shape() = Shape({piece_size, aligned_elem_cnt / target_piece_size});
}

void BoxingAlignBroadcastReshapeOp::InferBwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  const BoxingAlignBroadcastReshapeOpConf& conf = op_conf().boxing_align_broadcast_reshape_conf();
  const BlobDesc* out = GetBlobDesc4BnInOp("out");
  BlobDesc* reduce_buf = GetBlobDesc4BnInOp("reduce_buf");
  reduce_buf->mut_shape() = Shape({out->shape().elem_cnt()});
  reduce_buf->set_data_type(out->data_type());
  BlobDesc* bw_buf = GetBlobDesc4BnInOp("bw_buf");
  CHECK_EQ(out->shape().elem_cnt() % conf.target_parallel_num(), 0);
  bw_buf->mut_shape() = Shape({out->shape().elem_cnt() / conf.target_parallel_num()});
  bw_buf->set_data_type(out->data_type());
}

REGISTER_OP(OperatorConf::kBoxingAlignBroadcastReshapeConf, BoxingAlignBroadcastReshapeOp);

}  // namespace oneflow
