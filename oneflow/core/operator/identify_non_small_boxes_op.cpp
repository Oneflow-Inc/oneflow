#include "oneflow/core/operator/identify_non_small_boxes_op.h"

namespace oneflow {

void IdentifyNonSmallBoxesOp::InitFromOpConf() {
  CHECK(op_conf().has_identify_non_small_boxes_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& IdentifyNonSmallBoxesOp::GetCustomizedConf() const {
  return this->op_conf().identify_non_small_boxes_conf();
}

void IdentifyNonSmallBoxesOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in->shape().NumAxes(), 2);
  CHECK_EQ(in->shape().At(1), 4);
  CHECK(!in->has_instance_shape_field());
  // output
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({in->shape().At(0)});
  out->set_data_type(DataType::kInt8);
  if (in->has_dim0_valid_num_field()) {
    out->set_has_dim0_valid_num_field(true);
    out->mut_dim0_inner_shape() = Shape({1, out->shape().At(0)});
  }
}

void IdentifyNonSmallBoxesOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
}

REGISTER_OP(OperatorConf::kIdentifyNonSmallBoxesConf, IdentifyNonSmallBoxesOp);

}  // namespace oneflow
