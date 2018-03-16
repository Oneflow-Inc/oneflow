#include "oneflow/core/operator/elementwise_op.h"

namespace oneflow {

void ElementwiseOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out");
  VirtualInitFromOpConf();
}

void ElementwiseOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().at(0));
  for (size_t i = 1; i < input_bns().size(); ++i) {
    CHECK(*in_0_blob_desc == *GetBlobDesc4BnInOp(input_bns().at(i)));
  }
  *GetBlobDesc4BnInOp("out") = *in_0_blob_desc;
  VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

}  // namespace oneflow
