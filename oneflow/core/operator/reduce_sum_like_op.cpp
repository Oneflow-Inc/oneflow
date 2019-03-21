#include "oneflow/core/operator/reduce_sum_like_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void ReduceSumLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_like_conf());
  EnrollInputBn("x");
  EnrollInputBn("like")->set_use_header_only(true);
  EnrollOutputBn("y");
  EnrollFwBufBn("fw_tmp");
}

const PbMessage& ReduceSumLikeOp::GetCustomizedConf() const {
  return op_conf().reduce_sum_like_conf();
}

void ReduceSumLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  *GetBlobDesc4BnInOp("fw_tmp") = *GetBlobDesc4BnInOp("x");
  *GetBlobDesc4BnInOp("y") = *GetBlobDesc4BnInOp("like");
}

REGISTER_OP(OperatorConf::kReduceSumLikeConf, ReduceSumLikeOp);

}  // namespace oneflow
