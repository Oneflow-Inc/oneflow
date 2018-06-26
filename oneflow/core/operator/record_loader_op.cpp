#include "oneflow/core/operator/record_loader_op.h"

namespace oneflow {

void RecordLoaderOp::InitFromOpConf() {}

void RecordLoaderOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {}

}  // namespace oneflow
