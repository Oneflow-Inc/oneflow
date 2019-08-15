#include "oneflow/core/operator/maskrcnn_concat_uncontiguous_dim1_op.h"

namespace oneflow {

void MaskrcnnConcatUncontiguousDim1Op::InitFromOpConf() {
  CHECK(op_conf().has_maskrcnn_concat_uncontiguous_dim1_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& MaskrcnnConcatUncontiguousDim1Op::GetCustomizedConf() const {
  return this->op_conf().maskrcnn_concat_uncontiguous_dim1_conf();
}

void MaskrcnnConcatUncontiguousDim1Op::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK(in->has_dim1_valid_num_field());
  CHECK(!in->has_dim0_valid_num_field());
  CHECK(!in->has_instance_shape_field());
  CHECK(!in->has_dim2_valid_num_field());
  // output
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  auto dim_vec = in->shape().dim_vec();
  dim_vec[0] = dim_vec[0] * dim_vec[1];
  dim_vec.erase(dim_vec.begin() + 1);
  out->mut_shape() = Shape(dim_vec);
  out->set_data_type(in->data_type());
  out->set_has_dim0_valid_num_field(true);
  out->mut_dim0_inner_shape() = Shape({1, dim_vec[0]});
}

REGISTER_OP(OperatorConf::kMaskrcnnConcatUncontiguousDim1Conf, MaskrcnnConcatUncontiguousDim1Op);

}  // namespace oneflow
