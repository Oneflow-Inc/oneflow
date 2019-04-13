#include "oneflow/core/operator/reduce_sum_like_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void ReduceSumLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_like_conf());
  EnrollInputBn("x");
  EnrollInputBn("like")->set_use_header_only(true);
  EnrollOutputBn("y");
  EnrollOutputBn("temp_storage");
}

const PbMessage& ReduceSumLikeOp::GetCustomizedConf() const {
  return op_conf().reduce_sum_like_conf();
}

void ReduceSumLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  *GetBlobDesc4BnInOp("temp_storage") = *GetBlobDesc4BnInOp("x");
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
}

void ReduceSumLikeOp::GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>*) const {
  TODO();
}

REGISTER_OP(OperatorConf::kReduceSumLikeConf, ReduceSumLikeOp);

}  // namespace oneflow
