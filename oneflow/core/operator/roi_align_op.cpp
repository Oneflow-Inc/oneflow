#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RoiAlignOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoiAlignOp);
  RoiAlignOp() = default;
  ~RoiAlignOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().roi_align_conf(); }

  void InitFromOpConf() override {
    EnrollInputBn("x");
    EnrollInputBn("rois", false);
    EnrollOutputBn("y");
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const RoiAlignArgs& roi_align_args = op_conf().roi_align_conf().roi_align_args();
    if (roi_align_args.data_format() != "channels_first") { UNIMPLEMENTED(); }
    // x: feature map (N, C, H, W)
    const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
    CHECK_EQ(x_blob_desc->shape().NumAxes(), 4);
    // rois: (R, 5)
    const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
    CHECK_EQ(rois_blob_desc->shape().NumAxes(), 2);
    CHECK_EQ(rois_blob_desc->shape().At(1), 5);
    // y: (R, C, pool_h, pool_w)
    BlobDesc* y_blob_desc = GetBlobDesc4BnInOp("y");
    *y_blob_desc = *rois_blob_desc;
    y_blob_desc->mut_shape() = Shape({rois_blob_desc->shape().At(0), x_blob_desc->shape().At(1),
                                      roi_align_args.pooled_h(), roi_align_args.pooled_w()});
    y_blob_desc->set_data_type(x_blob_desc->data_type());
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("y") = *BatchAxis4BnInOp("rois");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("x", 0).Split("rois", 0).Split("y", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kRoiAlignConf, RoiAlignOp);

class RoiAlignGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoiAlignGradOp);
  RoiAlignGradOp() = default;
  ~RoiAlignGradOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().roi_align_grad_conf(); }

  void InitFromOpConf() override {
    EnrollInputBn("dy");
    EnrollInputBn("x_like")->set_use_header_only(true);
    EnrollInputBn("rois", false);
    EnrollOutputBn("dx");
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const RoiAlignArgs& roi_align_args = op_conf().roi_align_grad_conf().roi_align_args();
    if (roi_align_args.data_format() != "channels_first") { UNIMPLEMENTED(); }
    // x: feature map (N, C, H, W)
    const BlobDesc* x_like_blob_desc = GetBlobDesc4BnInOp("x_like");
    CHECK_EQ_OR_RETURN(x_like_blob_desc->shape().NumAxes(), 4);
    // rois: (R, 5)
    const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
    CHECK_EQ_OR_RETURN(rois_blob_desc->shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(rois_blob_desc->shape().At(1), 5);
    // y: (R, C, pool_h, pool_w)
    BlobDesc* dy_blob_desc = GetBlobDesc4BnInOp("dy");
    const Shape& y_shape = Shape({rois_blob_desc->shape().At(0), x_like_blob_desc->shape().At(1),
                                  roi_align_args.pooled_h(), roi_align_args.pooled_w()});
    CHECK_EQ_OR_RETURN(y_shape, dy_blob_desc->shape());
    CHECK_EQ_OR_RETURN(dy_blob_desc->data_type(), x_like_blob_desc->data_type());
    GetBlobDesc4BnInOp("dx")->CopyMetaFrom(*GetBlobDesc4BnInOp("x_like"));
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("dx") = *BatchAxis4BnInOp("x_like");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("dy", 0)
        .Split("rois", 0)
        .Split("x_like", 0)
        .Split("x", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kRoiAlignGradConf, RoiAlignGradOp);

}  // namespace oneflow
