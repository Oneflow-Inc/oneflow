#include "oneflow/core/operator/maskrcnn_positive_negative_sample_op.h"

namespace oneflow {

void MaskrcnnPositiveNegativeSampleOp::InitFromOpConf() {
  CHECK(op_conf().has_maskrcnn_positive_negative_sample_conf());
  EnrollInputBn("pos_inds", false);
  EnrollInputBn("neg_inds", false);
  EnrollOutputBn("sampled_pos_inds", false);
  EnrollOutputBn("sampled_neg_inds", false);
}

const PbMessage& MaskrcnnPositiveNegativeSampleOp::GetCustomizedConf() const {
  return this->op_conf().maskrcnn_positive_negative_sample_conf();
}

void MaskrcnnPositiveNegativeSampleOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto conf = op_conf().maskrcnn_positive_negative_sample_conf();
  // input: pos_inds (num_pos,)
  const BlobDesc* pos_inds = GetBlobDesc4BnInOp("pos_inds");
  CHECK_EQ(pos_inds->shape().NumAxes(), 1);
  // input: boxes (num_neg,)
  const BlobDesc* neg_inds = GetBlobDesc4BnInOp("neg_inds");
  CHECK_EQ(neg_inds->shape().NumAxes(), 1);
  // output: sampled_pos_inds (max_num_pos,)
  BlobDesc* sampled_pos_inds = GetBlobDesc4BnInOp("sampled_pos_inds");
  *sampled_pos_inds = *pos_inds;
  sampled_pos_inds->mut_shape() =
      Shape({static_cast<int64_t>(conf.total_subsample_num() * conf.pos_fraction())});
  sampled_pos_inds->set_has_dim0_valid_num_field(true);
  sampled_pos_inds->mut_dim0_inner_shape() = Shape({1, sampled_pos_inds->shape().At(0)});
  // output: sampled_neg_inds (max_num_neg,)
  BlobDesc* sampled_neg_inds = GetBlobDesc4BnInOp("sampled_neg_inds");
  *sampled_neg_inds = *neg_inds;
  sampled_neg_inds->mut_shape() = Shape({conf.total_subsample_num()});
  sampled_neg_inds->set_has_dim0_valid_num_field(true);
  sampled_neg_inds->mut_dim0_inner_shape() = Shape({1, sampled_neg_inds->shape().At(0)});
}

REGISTER_OP(OperatorConf::kMaskrcnnPositiveNegativeSampleConf, MaskrcnnPositiveNegativeSampleOp);

}  // namespace oneflow
