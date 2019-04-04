#include "oneflow/core/operator/keep_header_only_op.h"

namespace oneflow {

void KeepHeaderOnlyOp::InitFromOpConf() {
  CHECK_EQ(GetPbRpfFromCustomizedConf<std::string>("in").size(),
           GetPbRpfFromCustomizedConf<std::string>("out").size());
  EnrollRepeatedInputBn("in", false);
  EnrollRepeatedOutputBn("out", false);
}

void KeepHeaderOnlyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  size_t in_num = GetPbRpfFromCustomizedConf<std::string>("in").size();
  for (size_t i = 0; i < in_num; ++i) {
    BlobDesc* out = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    *out = *GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    out->set_is_body_disabled(true);
  }
}

REGISTER_OP(OperatorConf::kKeepHeaderOnlyConf, KeepHeaderOnlyOp);

}  // namespace oneflow
