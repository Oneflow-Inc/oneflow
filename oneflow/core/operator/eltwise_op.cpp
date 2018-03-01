#include "oneflow/core/operator/eltwise_op.h"

namespace oneflow {

void EltwiseOp::InitFromOpConf() {
  CHECK(op_conf().has_eltwise_conf());

  for (int i = 0; i < op_conf().eltwise_conf().in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    CHECK(ibn2lbn_.emplace(ibn, op_conf().eltwise_conf().in(i)).second);
    EnrollInputBn(ibn);
  }
  EnrollOutputBn("out");
}

const PbMessage& EltwiseOp::GetCustomizedConf() const {
  return op_conf().eltwise_conf();
}

void EltwiseOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // const EltwiseOpConf& conf = op_conf().eltwise_conf();
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
}

REGISTER_OP(OperatorConf::kEltwiseConf, EltwiseOp);

}  // namespace oneflow
