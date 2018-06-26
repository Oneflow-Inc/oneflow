#include "oneflow/core/operator/record_loader_op.h"

namespace oneflow {

void RecordLoaderOp::InitFromOpConf() {
  CHECK(op_conf().has_record_loader_conf());
  EnrollOutputBn("out");
}

void RecordLoaderOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const RecordLoaderOpConf& conf = op_conf().record_loader_conf();
  out_blob_desc->mut_shape() = Shape({conf.piece_size_in_each_loader()});
}

}  // namespace oneflow
