#include "oneflow/core/operator/split_like_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SplitLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_split_like_conf());
  EnrollInputBn("in");
  FOR_RANGE(int32_t, i, 0, op_conf().split_like_conf().like_size()) {
    EnrollInputBn(GenRepeatedBn("like", i), false)->set_use_header_only(true);
  }
  EnrollRepeatedOutputBn("out");
}

const PbMessage& SplitLikeOp::GetCustomizedConf() const { return op_conf().split_like_conf(); }

Maybe<void> SplitLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const SplitLikeOpConf& conf = op_conf().split_like_conf();
  const BlobDesc* like_0_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn("like", 0));
  const std::vector<int64_t>& in_dim_vec = GetBlobDesc4BnInOp("in")->shape().dim_vec();
  int32_t split_axis = FixAxis(conf.axis(), in_dim_vec.size());
  int64_t dim_sum = 0;
  FOR_RANGE(int32_t, i, 0, op_conf().split_like_conf().like_size()) {
    const BlobDesc* like_i_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn("like", i));
    FOR_RANGE(int64_t, j, 0, like_i_blob_desc->shape().NumAxes()) {
      if (j != split_axis) {
        CHECK_EQ_OR_RETURN(like_0_blob_desc->shape().dim_vec().at(j),
                           like_i_blob_desc->shape().At(j));
      }
    }
    dim_sum += like_i_blob_desc->shape().At(split_axis);
    BlobDesc* output_i_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    output_i_blob_desc->CopyMetaFrom(*like_i_blob_desc);
  }
  CHECK_EQ_OR_RETURN(dim_sum, in_dim_vec.at(split_axis));
  return Maybe<void>::Ok();
}

Maybe<void> SplitLikeOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp("in"); }
  return Maybe<void>::Ok();
}

Maybe<void> SplitLikeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const SplitLikeOpConf& conf = op_conf().split_like_conf();
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
  const int32_t axis = FixAxis(conf.axis(), num_axes);
  FOR_RANGE(int32_t, i, 0, num_axes) {
    if (i == axis) { continue; }
    SbpSignatureBuilder()
        .Split(input_bns(), i)
        .Split(output_bns(), i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

int32_t SplitLikeOp::FixAxis(const int32_t axis, const int64_t num_axes) const {
  int32_t ret = axis;
  if (axis < 0) { ret += num_axes; }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return ret;
}

REGISTER_OP(OperatorConf::kSplitLikeConf, SplitLikeOp);

}  // namespace oneflow
