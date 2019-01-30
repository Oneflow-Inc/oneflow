#include "oneflow/core/operator/half_to_float_op.h"

namespace oneflow {

void HalfToFloatOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void HalfToFloatOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  CHECK(in->data_type() == DataType::kFloat16);
  *out = *in;
  out->set_data_type(DataType::kFloat);
}

REGISTER_OP(OperatorConf::kHalfToFloatConf, HalfToFloatOp);

}  // namespace oneflow
