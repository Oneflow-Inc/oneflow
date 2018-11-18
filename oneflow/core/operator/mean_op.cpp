#include "oneflow/core/operator/mean_op.h"

namespace oneflow {

void MeanOp::InitFromOpConf() {
  CHECK(op_conf().has_mean_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollFwBufBn("fw_tmp");
  EnrollBwBufBn("bw_tmp");
}

void MeanOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  *out_blob = *in_blob;
  std::vector<int64_t> dim_vec = in_blob->shape().dim_vec();
  dim_vec.back() = 1;
  out_blob->mut_shape() = Shape(std::move(dim_vec));
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
}

void MeanOp::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  *GetBlobDesc4BnInOp("bw_tmp") = *GetBlobDesc4BnInOp("out");
}

REGISTER_OP(OperatorConf::kMeanConf, MeanOp);

}  // namespace oneflow
