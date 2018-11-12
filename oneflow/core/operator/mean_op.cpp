#include "oneflow/core/operator/mean_op.h"

namespace oneflow {

void MeanOp::InitFromOpConf() {
  CHECK(op_conf().has_mean_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void MeanOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob = GetBlobDesc4BnInOp("in");
  out_blob->mut_shape() = Shape({1});
}

REGISTER_OP(OperatorConf::kMeanConf, MeanOp);

}  // namespace oneflow
