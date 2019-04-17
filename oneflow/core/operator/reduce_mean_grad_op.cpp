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

void ReduceMeanGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  *GetBlobDesc4BnInOp("temp_storage") = *GetBlobDesc4BnInOp("dy");
  GetBlobDesc4BnInOp("dx")->CopyMetaFrom(*GetBlobDesc4BnInOp("x"));
}

void ReduceMeanGradOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  const auto& reduced_axes = op_conf().reduce_mean_grad_conf().reduced_axis();
  HashSet<int64_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  if (ReduceSbpUtil::IsReduceAxisSplitted(SbpInferHint4Ibn("x"), conf_axes) == false) {
    rules->emplace_back(MakeDataSplitSbpSignatureRule(this));
  }
  rules->emplace_back(MakeMultiIbnsBroadcastSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kReduceMeanGradConf, ReduceMeanGradOp);

}  // namespace oneflow
