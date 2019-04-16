#include "oneflow/core/operator/reduce_sum_like_op.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void ReduceSumLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_like_conf());
  EnrollInputBn("x");
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollOutputBn("y");
  EnrollOutputBn("temp_storage", false);
}

const PbMessage& ReduceSumLikeOp::GetCustomizedConf() const {
  return op_conf().reduce_sum_like_conf();
}

void ReduceSumLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  const ReduceSumLikeOpConf& conf = op_conf().reduce_sum_like_conf();
  BlobDesc* x_blob = GetBlobDesc4BnInOp("x");
  BlobDesc* like_blob = GetBlobDesc4BnInOp("like");
  if (conf.axis().empty()) { CHECK_EQ(x_blob->shape(), like_blob->shape()); }
  *GetBlobDesc4BnInOp("temp_storage") = *x_blob;
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*like_blob);
}

void ReduceSumLikeOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  const auto& reduced_axes = op_conf().reduce_sum_like_conf().axis();
  ReduceSbpUtil::GetReduceSumSplitSignatureRules(this, "x",
                                                 {reduced_axes.begin(), reduced_axes.end()}, rules);
  rules->emplace_back(MakeMultiIbnsBroadcastSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kReduceSumLikeConf, ReduceSumLikeOp);

}  // namespace oneflow
