#include "oneflow/core/operator/random_mask_like_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void RandomMaskLikeOp::InitFromOpConf() {
  if (op_conf().random_mask_like_conf().has_noise_shape()) { TODO(); }
  double rate = op_conf().random_mask_like_conf().rate();
  CHECK_GE(rate, 0);
  CHECK_LT(rate, 1);
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollTmpBn("random_tmp");
  EnrollOutputBn("out", false);
}

const PbMessage& RandomMaskLikeOp::GetCustomizedConf() const { return op_conf().random_mask_like_conf(); 

Maybe<void> RandomMaskLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // CHECK_EQ(op_conf().random_mask_like_conf().noise_shape().dim_size(),
  //          GetBlobDesc4BnInOp("in")->shape().NumAxes());
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("like");
  *GetBlobDesc4BnInOp("random_tmp") = *GetBlobDesc4BnInOp("like");
  GetBlobDesc4BnInOp("out")->set_data_type(DataType::kInt8);
  GetBlobDesc4BnInOp("random_tmp")->set_data_type(DataType::kFloat);
  return Maybe<void>::Ok();
}

Maybe<void> RandomMaskLikeOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp(SoleIbn()); }
  return Maybe<void>::Ok();
}

Maybe<void> RandomMaskLikeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn(SoleIbn()))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kRandomMaskLikeConf, RandomMaskLikeOp);

}  // namespace oneflow
