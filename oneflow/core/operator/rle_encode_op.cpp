#include "oneflow/core/operator/rle_encode_op.h"

namespace oneflow {

void RleEncodeOp::InitFromOpConf() {
  CHECK(op_conf().has_rle_encode_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& RleEncodeOp::GetCustomizedConf() const { return op_conf().rle_encode_conf(); }

void RleEncodeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  // input: in (R, H, W)
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in->data_type(), DataType::kUInt8);
  CHECK_EQ(in->shape().NumAxes(), 3);
  CHECK(in->has_dim0_valid_num_field());
  CHECK(in->has_instance_shape_field());
  // output: out(R, RLE_MAX_BYTES)
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape() = Shape({in->shape().At(0), op_conf().rle_encode_conf().rle_max_bytes()});
  out->set_has_dim1_valid_num_field(true);
  out->set_has_instance_shape_field(false);
  out->set_data_type(DataType::kChar);
}

REGISTER_CPU_OP(OperatorConf::kRleEncodeConf, RleEncodeOp);

}  // namespace oneflow
