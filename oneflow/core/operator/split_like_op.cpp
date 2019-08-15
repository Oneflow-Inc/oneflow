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

void SplitLikeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  const SplitLikeOpConf& conf = op_conf().split_like_conf();
  const BlobDesc* like_0_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn("like", 0));
  int32_t split_axis = conf.axis();
  std::vector<int64_t> in_dim_vec = GetBlobDesc4BnInOp("in")->shape().dim_vec();
  if (split_axis < 0) { split_axis += in_dim_vec.size(); }
  CHECK_GE(split_axis, 0);
  int64_t dim_sum = 0;
  for (size_t i = 0; i < op_conf().split_like_conf().like_size(); ++i) {
    const BlobDesc* like_i_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn("like", i));
    for (int64_t j = 0; j < like_i_blob_desc->shape().NumAxes(); ++j) {
      if (j != split_axis) {
        CHECK_EQ(like_0_blob_desc->shape().dim_vec().at(j), like_i_blob_desc->shape().At(j));
      }
    }
    dim_sum += like_i_blob_desc->shape().At(split_axis);
    BlobDesc* output_i_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    output_i_blob_desc->set_data_type(like_i_blob_desc->data_type());
    output_i_blob_desc->mut_shape() = like_i_blob_desc->shape();
  }
  CHECK_EQ(dim_sum, in_dim_vec.at(split_axis));
}

void SplitLikeOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const SplitLikeOpConf& conf = op_conf().split_like_conf();
  const int axis = conf.axis();
  const int num_axes = LogicalBlobDesc4Ibn("in").shape().NumAxes();
  for (int i = 0; i < num_axes; ++i) {
    if (i == axis) { continue; }
    SbpSignatureBuilder()
        .Split(output_bns(), i)
        .Split(output_bns(), i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
}

REGISTER_OP(OperatorConf::kSplitLikeConf, SplitLikeOp);

}  // namespace oneflow
