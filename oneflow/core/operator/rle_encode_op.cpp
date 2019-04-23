#include "oneflow/core/operator/rle_encode_op.h"

namespace oneflow {

void RleEncodeOp::InitFromOpConf() {
  CHECK(op_conf().has_rle_encode_conf());
  EnrollInputBn("in", false);
  EnrollInputBn("size", false);
  EnrollOutputBn("out", false);
}

const PbMessage& RleEncodeOp::GetCustomizedConf() const { return op_conf().rle_encode_conf(); }

void RleEncodeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  // input: in (R, H, W)
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in->data_type(), DataType::kUInt8);
  CHECK_EQ(in->shape().NumAxes(), 3);
  CHECK(in->has_record_id_in_device_piece_field());
  // input: size (N, 2)
  const BlobDesc* size = GetBlobDesc4BnInOp("size");
  CHECK_EQ(size->data_type(), DataType::kInt32);
  CHECK_EQ(size->shape().NumAxes(), 2);
  CHECK_EQ(size->shape().At(1), 2);
  // output: out(R, RLE_MAX_BYTES)
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(DataType::kChar);
  out->mut_shape() = Shape({in->shape().At(0), op_conf().rle_encode_conf().rle_max_bytes()});
  out->mut_dim0_inner_shape() = in->dim0_inner_shape();
  out->set_has_dim0_valid_num_field(in->has_dim0_valid_num_field());
  out->set_has_dim1_valid_num_field(true);
  out->set_has_record_id_in_device_piece_field(in->has_record_id_in_device_piece_field());
}

REGISTER_CPU_OP(OperatorConf::kRleEncodeConf, RleEncodeOp);

}  // namespace oneflow
