#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RandomPermLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomPermLikeOp);
  RandomPermLikeOp() = default;
  ~RandomPermLikeOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_random_perm_like_conf());
    const RandomPermLikeOpConf& random_perm_like_conf = this->op_conf().random_perm_like_conf();
    EnrollInputBn("like", false)->set_use_header_only(true);
    EnrollTmpBn("random_mask");
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().random_perm_like_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    BlobDesc* random_mask_desc = GetBlobDesc4BnInOp("random_mask");
    const BlobDesc* like_blob_desc = GetBlobDesc4BnInOp("like");
    out_blob_desc->set_is_dynamic(like_blob_desc->is_dynamic());
    out_blob_desc->mut_shape() = Shape({like_blob_desc->shape().At(0)});
    out_blob_desc->set_data_type(DataType::kInt32);
    *random_mask_desc = *out_blob_desc;
    random_mask_desc->set_data_type(DataType::kFloat);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    BatchAxis4BnInOp("out")->set_value(0);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kRandomPermLikeConf, RandomPermLikeOp);

}  // namespace oneflow
