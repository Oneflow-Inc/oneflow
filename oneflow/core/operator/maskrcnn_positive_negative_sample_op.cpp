#include "oneflow/core/operator/operator.h"

namespace oneflow {

class MaskrcnnPositiveNegativeSampleOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskrcnnPositiveNegativeSampleOp);
  MaskrcnnPositiveNegativeSampleOp() = default;
  ~MaskrcnnPositiveNegativeSampleOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_maskrcnn_positive_negative_sample_conf());
    EnrollInputBn("pos_inds", false);
    EnrollInputBn("neg_inds", false);
    EnrollOutputBn("sampled_pos_inds", false);
    EnrollOutputBn("sampled_neg_inds", false);
  }
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().maskrcnn_positive_negative_sample_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const auto conf = op_conf().maskrcnn_positive_negative_sample_conf();
    // input: pos_inds (num_pos,)
    const BlobDesc* pos_inds = GetBlobDesc4BnInOp("pos_inds");
    CHECK_EQ_OR_RETURN(pos_inds->shape().NumAxes(), 1);
    // input: boxes (num_neg,)
    const BlobDesc* neg_inds = GetBlobDesc4BnInOp("neg_inds");
    CHECK_EQ_OR_RETURN(neg_inds->shape().NumAxes(), 1);
    // output: sampled_pos_inds (max_num_pos,)
    BlobDesc* sampled_pos_inds = GetBlobDesc4BnInOp("sampled_pos_inds");
    *sampled_pos_inds = *pos_inds;
    sampled_pos_inds->mut_shape() =
        Shape({static_cast<int64_t>(conf.total_subsample_num() * conf.pos_fraction())});
    sampled_pos_inds->set_is_dynamic(true);
    // output: sampled_neg_inds (max_num_neg,)
    BlobDesc* sampled_neg_inds = GetBlobDesc4BnInOp("sampled_neg_inds");
    *sampled_neg_inds = *neg_inds;
    sampled_neg_inds->mut_shape() = Shape({conf.total_subsample_num()});
    sampled_neg_inds->set_is_dynamic(true);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    BatchAxis4BnInOp("sampled_pos_inds")->set_value(0);
    BatchAxis4BnInOp("sampled_neg_inds")->set_value(0);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kMaskrcnnPositiveNegativeSampleConf, MaskrcnnPositiveNegativeSampleOp);

}  // namespace oneflow
