#include "oneflow/core/operator/add_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

bool IsAllInputPartialSumParallel(
    const Operator& op,
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn) {
  for (const auto& ibn : op.input_bns()) {
    if (SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel() == false) { return false; }
  }
  return true;
}

bool IsAllInputBroadcastParallel(
    const Operator& op,
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn) {
  for (const auto& ibn : op.input_bns()) {
    if (SbpInferHint4Ibn(ibn).sbp_parallel().has_broadcast_parallel() == false) { return false; }
  }
  return true;
}

void AddOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_conf()); }
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }
void AddOp::VirtualFixInDiffBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  if (!Global<JobDesc>::Get()->enable_blob_mem_sharing()) { return; }
  int64_t blob_mem_id = oneflow_cast<int64_t>(NewUniqueId());
  FOR_RANGE(size_t, i, 0, input_diff_bns().size()) {
    GetBlobDesc4BnInOp(input_diff_bns().Get(i))->set_blob_mem_id(blob_mem_id);
  }
}

void AddOp::InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  for (const auto& ibn : input_bns()) {
    CHECK_EQ(*HasBatchDim4BnInOp(ibn), *HasBatchDim4BnInOp(input_bns().Get(0)));
  }
  NaiveInferHasBatchDim(HasBatchDim4BnInOp);
}

void AddOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int64_t num_axes = LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes();
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(sbp_sig_list);
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
