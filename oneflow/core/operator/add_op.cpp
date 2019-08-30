#include "oneflow/core/operator/add_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void AddOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_conf()); }
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }

Maybe<void> AddOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  for (const auto& ibn : input_bns()) {
    CHECK_EQ(*HasBatchDim4BnInOp(ibn), *HasBatchDim4BnInOp(input_bns().Get(0)));
  }
  return NaiveInferHasBatchDim(HasBatchDim4BnInOp);
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
