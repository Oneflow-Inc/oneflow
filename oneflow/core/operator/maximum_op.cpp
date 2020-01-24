#include "oneflow/core/operator/maximum_op.h"

namespace oneflow {

void MaximumOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_maximum_conf());
  EnrollTmpBn("mask");
}

const PbMessage& MaximumOp::GetCustomizedConf() const { return op_conf().maximum_conf(); }

Maybe<void> MaximumOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  BlobDesc* mask_blob_desc = GetBlobDesc4BnInOp("mask");
  *mask_blob_desc = *in_0_blob_desc;
  mask_blob_desc->set_data_type(DataType::kInt32);
  return Maybe<void>::Ok();
}

Maybe<void> MaximumOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& ibn : input_bns()) {
    CHECK_OR_RETURN(*BatchAxis4BnInOp(ibn) == *BatchAxis4BnInOp(input_bns().Get(0)));
  }
  return NaiveInferBatchAxis(BatchAxis4BnInOp);
}

Maybe<void> MaximumOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int64_t num_axes = JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes();
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(sbp_sig_list);
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kMaximumConf, MaximumOp);

}  // namespace oneflow
