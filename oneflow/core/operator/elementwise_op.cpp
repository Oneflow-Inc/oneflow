#include "oneflow/core/operator/elementwise_op.h"

namespace oneflow {

void ElementwiseOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
}

void ElementwiseOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().at(0));
  std::vector<int64_t> out_dim_vec = in_0_blob_desc->shape().dim_vec();
  for (size_t i = 1; i < input_bns().size(); ++i) {
    CHECK(*in_0_blob_desc == *GetBlobDesc4BnInOp(input_bns().at(i)));
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_0_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
}

}  // namespace oneflow
