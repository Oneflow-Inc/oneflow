#include "oneflow/core/operator/smooth_l1_op.h"

namespace oneflow {

void SmoothL1Op::InitFromOpConf() {
  CHECK(op_conf().has_smooth_l1_conf());
  EnrollInputBn("prediction");
  EnrollInputBn("label");
  EnrollInputBn("inside_weights");
  EnrollInputBn("out_weights");
  EnrollOutputBn("out");
}

const PbMessage& SmoothL1Op::GetCustomizedConf() const { return op_conf().smooth_l1_conf(); }

void SmoothL1Op::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* prediction = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label = GetBlobDesc4BnInOp("label");
  const BlobDesc* inside_weights = GetBlobDesc4BnInOp("inside_weights");
  const BlobDesc* outside_weights = GetBlobDesc4BnInOp("outside_weights");

  CHECK_EQ(prediction->shape(), label->shape());
  CHECK_EQ(prediction->data_type(), label->data_type());
  CHECK_EQ(prediction->shape(), inside_weights->shape());
  CHECK_EQ(prediction->shape(), outside_weights->shape());
  CHECK_GE(prediction->shape().NumAxes(), 2);

  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({prediction->shape()});
  out->set_data_type(prediction->data_type());
  out->set_has_data_id_field(prediction->has_data_id_field());
}

REGISTER_OP(OperatorConf::kSmoothL1Conf, SmoothL1Op);

}  // namespace oneflow
