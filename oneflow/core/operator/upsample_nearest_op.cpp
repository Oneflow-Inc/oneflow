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

};  // namespace oneflow

REGISTER_OP(OperatorConf::kUpsampleNearestConf, UpsampleNearestOp);

}  // namespace oneflow
