#include "oneflow/core/operator/record_loader_op.h"

namespace oneflow {

void RecordLoaderOp::InitFromOpConf() {
  CHECK(op_conf().has_record_loader_conf());
  EnrollOutputBn("out");
}

void RecordLoaderOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  int64_t global_piece_size = Global<JobDesc>::Get()->PieceSize();
  CHECK_EQ(global_piece_size % parallel_ctx->parallel_num(), 0);
  out_blob_desc->mut_shape() = Shape({global_piece_size / parallel_ctx->parallel_num()});
}

}  // namespace oneflow
