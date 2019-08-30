#include "oneflow/core/operator/reduce_mean_grad_op.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

void ReduceMeanGradOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_mean_grad_conf());
  EnrollInputBn("dy");
  EnrollInputBn("x", false)->set_use_header_only(true);
  EnrollOutputBn("dx");
  EnrollOutputBn("temp_storage");
}

const PbMessage& ReduceMeanGradOp::GetCustomizedConf() const {
  return op_conf().reduce_mean_grad_conf();
}

Maybe<void> ReduceMeanGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  *GetBlobDesc4BnInOp("temp_storage") = *GetBlobDesc4BnInOp("dy");
  GetBlobDesc4BnInOp("dx")->CopyMetaFrom(*GetBlobDesc4BnInOp("x"));
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReduceMeanGradConf, ReduceMeanGradOp);

}  // namespace oneflow
