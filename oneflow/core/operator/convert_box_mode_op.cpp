#include "oneflow/core/operator/convert_box_mode_op.h"

namespace oneflow {

void ConvertBoxModeOp::InitFromOpConf() {
  CHECK(op_conf().has_convert_box_mode_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& ConvertBoxModeOp::GetCustomizedConf() const {
  return this->op_conf().convert_box_mode_conf();
}

void ConvertBoxModeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  std::string mode = op_conf().convert_box_mode_conf().mode();
  CHECK(mode == "xyxy" || mode == "xywh");
  // input: (N, 4)
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in->shape().NumAxes(), 2);
  CHECK_EQ(in->shape().At(1), 4);
  CHECK(!in->has_instance_shape_field());
  // output: (N, 4)
  *GetBlobDesc4BnInOp("boxes") = *in;
}

REGISTER_OP(OperatorConf::kConvertBoxModeConf, ConvertBoxModeOp);

}  // namespace oneflow
