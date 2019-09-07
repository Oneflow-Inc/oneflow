#include "oneflow/core/operator/cwise_op.h"

namespace oneflow {

void CWiseOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in_0");
  VirtualInitFromOpConf();
}

Maybe<void> CWiseOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const auto* blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i));
    CHECK_OR_RETURN(*in_0_blob_desc == *blob_desc);
  }
  *GetBlobDesc4BnInOp("out") = *in_0_blob_desc;
  return VirtualInferBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

}  // namespace oneflow
