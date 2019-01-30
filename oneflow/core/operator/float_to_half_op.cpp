#include "oneflow/core/operator/float_to_half_op.h"

namespace oneflow {

void FloatToHalfOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void FloatToHalfOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  CHECK(in->data_type() == DataType::kFloat);
  out->set_data_type(DataType::kFloat16);
}

REGISTER_OP(OperatorConf::kFloatToHalfConf, FloatToHalfOp);

}  // namespace oneflow
