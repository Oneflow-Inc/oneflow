#include "oneflow/core/operator/local_reponse_normalization_op.h"

namespace oneflow {

void LocalResponseNormalizationOp::InitFromOpConf() {
  // TODO
}

const PbMessage& LocalResponseNormalizationOp::GetCustomizedConf() const {
  return op_conf().local_response_normalization_conf();
}

void LocalResponseNormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // TODO
}

}  // namespace oneflow
