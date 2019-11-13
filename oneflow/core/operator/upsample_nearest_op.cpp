#include "oneflow/core/operator/operator.h"

namespace oneflow {

class UpsampleNearestOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UpsampleNearestOp);
  UpsampleNearestOp() = default;
  ~UpsampleNearestOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_upsample_nearest_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().upsample_nearest_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    if (op_conf().upsample_nearest_conf().data_format() != "channels_first"
        || in_blob_desc->shape().NumAxes() != 4) {
      LOG(FATAL) << "upsample_nearest only supports NCHW";
    }
    const int32_t scale = op_conf().upsample_nearest_conf().scale();
    CHECK_GT(scale, 1);
    out_blob_desc->mut_shape() =
        Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
               scale * in_blob_desc->shape().At(2), scale * in_blob_desc->shape().At(3)});
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kUpsampleNearestConf, UpsampleNearestOp);

class UpsampleNearestGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UpsampleNearestGradOp);
  UpsampleNearestGradOp() = default;
  ~UpsampleNearestGradOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_upsample_nearest_grad_conf());
    EnrollInputBn("dy");
    EnrollOutputBn("dx");
  }

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().upsample_nearest_grad_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("dy");
    const BlobDesc* dy_blob_desc = GetBlobDesc4BnInOp("dy");
    BlobDesc* dx_blob_desc = GetBlobDesc4BnInOp("dx");
    const int32_t scale = op_conf().upsample_nearest_grad_conf().scale();
    CHECK_GT(scale, 1);
    dx_blob_desc->mut_shape() =
        Shape({dy_blob_desc->shape().At(0), dy_blob_desc->shape().At(1),
               dy_blob_desc->shape().At(2) / scale, dy_blob_desc->shape().At(3) / scale});
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("dx") = *BatchAxis4BnInOp("dy");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("dy", 0).Split("dx", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kUpsampleNearestGradConf, UpsampleNearestGradOp);

}  // namespace oneflow
