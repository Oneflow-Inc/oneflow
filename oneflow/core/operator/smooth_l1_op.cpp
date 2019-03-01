#include "oneflow/core/operator/smooth_l1_op.h"

namespace oneflow {

void SmoothL1Op::InitFromOpConf() {
  CHECK(op_conf().has_smooth_l1_conf());
  EnrollInputBn("prediction");
  EnrollInputBn("label");
  EnrollOutputBn("out");
}

const PbMessage& SmoothL1Op::GetCustomizedConf() const { return op_conf().smooth_l1_conf(); }

void SmoothL1Op::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const BlobDesc* prediction = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label = GetBlobDesc4BnInOp("label");
  CHECK_EQ(prediction->shape(), label->shape());
  CHECK_EQ(prediction->data_type(), label->data_type());
  CHECK_GE(prediction->shape().NumAxes(), 2);
  CHECK_GE(op_conf().smooth_l1_conf().beta(), 0);

  // out
  *GetBlobDesc4BnInOp("out") = *prediction;
}

REGISTER_OP(OperatorConf::kSmoothL1Conf, SmoothL1Op);

}  // namespace oneflow
