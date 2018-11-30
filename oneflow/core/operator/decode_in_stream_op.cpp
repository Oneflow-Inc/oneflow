#include "oneflow/core/operator/decode_in_stream_op.h"

namespace oneflow {

void DecodeInStreamOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_in_stream_conf());
  EnrollOutputBn("out", false);
}

const PbMessage& DecodeInStreamOp::GetCustomizedConf() const {
  return op_conf().decode_in_stream_conf();
}

void DecodeInStreamOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  // op_conf().decode_in_stream_conf()
  const DecodeInStreamOpConf& conf = op_conf().decode_in_stream_conf();
  std::vector<int64_t> dim_vec(1 + conf.shape().dim_size());
  int64_t global_piece_size = Global<JobDesc>::Get()->PieceSize();
  CHECK_EQ(global_piece_size % parallel_ctx->parallel_num(), 0);
  dim_vec[0] = global_piece_size / parallel_ctx->parallel_num();
  FOR_RANGE(size_t, j, 1, dim_vec.size()) { dim_vec[j] = conf.shape().dim(j - 1); }
  out_blob_desc->mut_shape() = Shape(dim_vec);
  out_blob_desc->set_data_type(conf.data_type());
  out_blob_desc->set_has_data_id_field(Global<JobDesc>::Get()->SizeOfOneDataId() > 0);
  bool r = conf.has_dim0_inner_shape();
  out_blob_desc->set_has_dim0_valid_num_field(r);
  if (conf.has_dim0_inner_shape()) {
    out_blob_desc->mut_dim0_inner_shape() = Shape(conf.dim0_inner_shape());
  }
  if (conf.has_dim0_valid_num()) { CHECK(conf.has_dim0_inner_shape()); }
}

REGISTER_OP(OperatorConf::kDecodeInStreamConf, DecodeInStreamOp);

}  // namespace oneflow