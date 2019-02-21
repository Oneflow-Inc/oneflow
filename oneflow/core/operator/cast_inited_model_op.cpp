#include "oneflow/core/operator/cast_inited_model_op.h"

namespace oneflow {

void CastInitedModelOp::InitFromOpConf() {
  EnrollInputBn("half_model");
  EnrollOutputBn("float_model");
}

void CastInitedModelOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {}

REGISTER_OP(OperatorConf::kCastInitedModelConf, CastInitedModelOp);

}  // namespace oneflow
