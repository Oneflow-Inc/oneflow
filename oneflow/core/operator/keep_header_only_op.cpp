#include "oneflow/core/operator/keep_header_only_op.h"

namespace oneflow {

void KeepHeaderOnlyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *GetBlobDesc4BnInOp("in");
  out->set_decouple_header_and_body(true);
}

REGISTER_OP(OperatorConf::kKeepHeaderOnlyConf, KeepHeaderOnlyOp);

}
