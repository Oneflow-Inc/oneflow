#include "oneflow/core/operator/operator.h"

namespace oneflow {

class MasksCropAndResizeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MasksCropAndResizeOp);
  MasksCropAndResizeOp() = default;
  ~MasksCropAndResizeOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_masks_crop_and_resize_conf());
    EnrollInputBn("masks", false);
    EnrollInputBn("rois", false);
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().masks_crop_and_resize_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const auto& conf = op_conf().masks_crop_and_resize_conf();
    // input: masks (N, C, H, W)
    const BlobDesc* masks = GetBlobDesc4BnInOp("masks");
    CHECK_EQ(masks->shape().NumAxes(), 4);
    CHECK_EQ(masks->is_tensor_list(), false);
    CHECK_EQ(masks->data_type(), DataType::kInt8);
    // input: rois (N, 4)
    const BlobDesc* rois = GetBlobDesc4BnInOp("rois");
    CHECK_EQ(rois->shape().NumAxes(), 2);
    CHECK_EQ(rois->shape().At(1), 4);
    CHECK_EQ(rois->is_tensor_list(), false);
    CHECK_EQ(masks->shape().At(0), rois->shape().At(0));
    CHECK_EQ(masks->is_dynamic(), rois->is_dynamic());
    // output: (N, C, mask_h, mask_w)
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->CopyMetaFrom(*rois);
    out->mut_shape() =
        Shape({masks->shape().At(0), masks->shape().At(1), conf.mask_height(), conf.mask_width()});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("masks", 0)
        .Split("rois", 0)
        .Split("out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kMasksCropAndResizeConf, MasksCropAndResizeOp);

}  // namespace oneflow
