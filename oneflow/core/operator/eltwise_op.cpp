#include "oneflow/core/operator/eltwise_op.h"

namespace oneflow {

void EltwiseOp::InitFromOpConf() {
  CHECK(op_conf().has_eltwise_conf());

  for (int i = 0; i < op_conf().eltwise_conf().in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    EnrollInputBn(ibn);
  }
  EnrollOutputBn("out");
  EnrollOutputBn("tmp");
}

const PbMessage& EltwiseOp::GetCustomizedConf() const {
  return op_conf().eltwise_conf();
}

void EltwiseOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().at(0));
  std::vector<int64_t> out_dim_vec = in_0_blob_desc->shape().dim_vec();
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const BlobDesc* in_i_blob_desc = GetBlobDesc4BnInOp(input_bns().at(i));
    CHECK_EQ(in_i_blob_desc->data_type(), in_0_blob_desc->data_type());
    CHECK_EQ(in_i_blob_desc->has_data_id_field(),
             in_0_blob_desc->has_data_id_field());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_0_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
  // this blob is for GPU implementation
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("tmp");
  *tmp_blob_desc = *in_0_blob_desc;
  tmp_blob_desc->mut_shape() = Shape(out_dim_vec);
  tmp_blob_desc->set_has_data_id_field(false);
  // this blob is for mask recording in blob index of a max value
  BlobDesc* mask_blob_desc = GetBlobDesc4BnInOp("mask");
  *mask_blob_desc = *in_0_blob_desc;
  mask_blob_desc->mut_shape() = Shape(out_dim_vec);
  mask_blob_desc->set_has_data_id_field(false);
}

REGISTER_OP(OperatorConf::kEltwiseConf, EltwiseOp);

}  // namespace oneflow
