#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RandomPermOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomPermOp);
  RandomPermOp() = default;
  ~RandomPermOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_random_perm_conf());
    const RandomPermOpConf& random_perm_conf = this->op_conf().random_perm_conf();
    if (random_perm_conf.has_like()) { EnrollInputBn("like", false)->set_use_header_only(true); }
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return this->op_conf().random_perm_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const RandomPermOpConf& random_perm_conf = this->op_conf().random_perm_conf();
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    int64_t out_size = -1;
    if (random_perm_conf.has_like()) {
      const BlobDesc* like_blob_desc = GetBlobDesc4BnInOp("like");
      out_blob_desc->set_is_dynamic(like_blob_desc->is_dynamic());
      out_size = like_blob_desc->shape().At(0);
    } else if (random_perm_conf.has_upper_bound()) {
      out_size = random_perm_conf.upper_bound();
    }
    CHECK_GE_OR_RETURN(out_size, 0);
    out_blob_desc->mut_shape() = Shape({out_size});
    out_blob_desc->set_data_type(DataType::kInt32);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    BatchAxis4BnInOp("out")->set_value(0);
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kRandomPermConf, RandomPermOp);

}  // namespace oneflow
