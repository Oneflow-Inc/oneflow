#include "oneflow/core/operator/v_stack_op.h"

namespace oneflow {

void VStackOp::InitFromOpConf() {
  CHECK(op_conf().has_v_stack_conf());
  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& VStackOp::GetCustomizedConf() const { return op_conf().v_stack_conf(); }

void VStackOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const VStackOpConf& conf = op_conf().v_stack_conf();
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  std::vector<int64_t> out_dim_vec = in_0_blob_desc->shape().dim_vec();
  auto CheckShape = [in_0_blob_desc](const Shape& shape) {
    int64_t in_axes_num = in_0_blob_desc->shape().NumAxes();
    CHECK_EQ(shape.NumAxes(), in_axes_num);
    for (int64_t j = 1; j < in_axes_num; ++j) {
      CHECK_EQ(in_0_blob_desc->shape().At(j), shape.At(j));
    }
  };
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const BlobDesc* in_i_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    CheckShape(in_i_blob_desc->shape());
    CHECK_EQ(in_i_blob_desc->data_type(), in_0_blob_desc->data_type());
    CHECK_EQ(in_i_blob_desc->has_data_id_field(), in_0_blob_desc->has_data_id_field());
    if (conf.identity_shape()) {
      CHECK(in_i_blob_desc->has_varing_instance_num());
      CHECK_EQ(in_i_blob_desc->shape().At(0), in_0_blob_desc->shape().At(0));
    } else {
      CHECK(!in_i_blob_desc->has_varing_instance_num());
      out_dim_vec[0] += in_i_blob_desc->shape().At(0);
    }
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_0_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
}

REGISTER_OP(OperatorConf::kVStackConf, VStackOp);

}  // namespace oneflow
