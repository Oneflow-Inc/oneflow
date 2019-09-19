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
    const RoiAlignConf& roi_align_conf = op_conf().roi_align_conf().roi_align_conf();
    if (roi_align_conf.data_format() != "channels_first") { UNIMPLEMENTED(); }
    // in: feature map (N, C, H, W)
    const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
    CHECK_EQ(x_blob_desc->shape().NumAxes(), 4);
    // rois: (R, 5)
    const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
    CHECK_EQ(rois_blob_desc->shape().NumAxes(), 2);
    CHECK_EQ(rois_blob_desc->shape().At(1), 5);
    CHECK(rois_blob_desc->has_dim0_inner_shape());
    // out: (R, C, pool_h, pool_w)
    BlobDesc* y_blob_desc = GetBlobDesc4BnInOp("y");
    *y_blob_desc = *rois_blob_desc;
    y_blob_desc->mut_shape() = Shape({rois_blob_desc->shape().At(0), x_blob_desc->shape().At(1),
                                      roi_align_conf.pooled_h(), roi_align_conf.pooled_w()});
    y_blob_desc->set_data_type(x_blob_desc->data_type());
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    oneflow::OptInt64* x_split_axis = BatchAxis4BnInOp("x");
    if (x_split_axis->has_value()) { CHECK_EQ_OR_RETURN(x_split_axis->value(), 0); }
    oneflow::OptInt64* roi_split_axis = BatchAxis4BnInOp("rois");
    if (roi_split_axis->has_value()) { CHECK_EQ_OR_RETURN(x_split_axis->value(), 0); }
    *BatchAxis4BnInOp("y") = *BatchAxis4BnInOp("x");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("x", 0).Split("y", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kRoiAlignConf, RoiAlignOp);

}  // namespace oneflow
