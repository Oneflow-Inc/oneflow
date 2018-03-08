#include "oneflow/core/operator/maximum_op.h"

namespace oneflow {

void MaximumOp::InitFromOpConf() {
  CHECK(op_conf().has_maximum_conf());
  ElementwiseOp::InitFromOpConf();
  EnrollOutputBn("mask");
}

const PbMessage& MaximumOp::GetCustomizedConf() const {
  return op_conf().maximum_conf();
}

void MaximumOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  ElementwiseOp::InferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().at(0));
  BlobDesc* mask_blob_desc = GetBlobDesc4BnInOp("mask");
  *mask_blob_desc = *in_0_blob_desc;
  mask_blob_desc->mut_shape() = Shape(in_0_blob_desc->shape().dim_vec());
  mask_blob_desc->set_has_data_id_field(false);
}

REGISTER_OP(OperatorConf::kMaximumConf, MaximumOp);

}  // namespace oneflow
