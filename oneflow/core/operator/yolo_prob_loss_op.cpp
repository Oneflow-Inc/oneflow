#include "oneflow/core/operator/operator.h"

namespace oneflow {

class YoloProbLossOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloProbLossOp);
  YoloProbLossOp() = default;
  ~YoloProbLossOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_yolo_prob_loss_conf());
    // Enroll input
    EnrollInputBn("bbox_objness");
    EnrollInputBn("bbox_clsprob");
    EnrollInputBn("pos_cls_label", false);
    EnrollInputBn("pos_inds", false);
    EnrollInputBn("neg_inds", false);
    EnrollInputBn("valid_num", false);

    // Enroll output
    EnrollOutputBn("bbox_objness_out", true);
    EnrollOutputBn("bbox_clsprob_out", true);
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().yolo_prob_loss_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // input: bbox_objness : (n, r, 1)  r = h*w*3
    const BlobDesc* bbox_objness_blob_desc = GetBlobDesc4BnInOp("bbox_objness");
    // input: bbox_clsprob : (n, r, 80)  r = h*w*3
    const BlobDesc* bbox_clsprob_blob_desc = GetBlobDesc4BnInOp("bbox_clsprob");
    // input: pos_cls_label (n, r)
    const BlobDesc* pos_cls_label_blob_desc = GetBlobDesc4BnInOp("pos_cls_label");
    // input: pos_inds (n, r) int32_t
    const BlobDesc* pos_inds_blob_desc = GetBlobDesc4BnInOp("pos_inds");
    // input: neg_inds (n, r) int32_t
    const BlobDesc* neg_inds_blob_desc = GetBlobDesc4BnInOp("neg_inds");

    const int64_t num_images = bbox_objness_blob_desc->shape().At(0);
    CHECK_EQ(num_images, bbox_clsprob_blob_desc->shape().At(0));
    CHECK_EQ(num_images, pos_cls_label_blob_desc->shape().At(0));
    CHECK_EQ(num_images, pos_inds_blob_desc->shape().At(0));
    CHECK_EQ(num_images, neg_inds_blob_desc->shape().At(0));
    const int64_t num_boxes = bbox_objness_blob_desc->shape().At(1);
    const int64_t num_clsprobs = op_conf().yolo_prob_loss_conf().num_classes();
    CHECK_EQ(num_boxes, pos_cls_label_blob_desc->shape().At(1));
    CHECK_EQ(num_boxes, pos_inds_blob_desc->shape().At(1));
    CHECK_EQ(num_boxes, neg_inds_blob_desc->shape().At(1));
    CHECK_EQ(1, bbox_objness_blob_desc->shape().At(2));
    CHECK_EQ(num_clsprobs, bbox_clsprob_blob_desc->shape().At(2));

    // output: bbox_objness_out (n, r, 1)
    *GetBlobDesc4BnInOp("bbox_objness_out") = *bbox_objness_blob_desc;

    // output: bbox_clsprob_out (n, r, 80)
    *GetBlobDesc4BnInOp("bbox_clsprob_out") = *bbox_clsprob_blob_desc;

    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("bbox_objness", 0)
        .Split("bbox_clsprob", 0)
        .Split("pos_cls_label", 0)
        .Split("pos_inds", 0)
        .Split("neg_inds", 0)
        .Split("valid_num", 0)
        .Split("bbox_objness_out", 0)
        .Split("bbox_clsprob_out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());

    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("bbox_objness_out") = *BatchAxis4BnInOp("bbox_objness");
    *BatchAxis4BnInOp("bbox_clsprob_out") = *BatchAxis4BnInOp("bbox_objness");
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kYoloProbLossConf, YoloProbLossOp);

}  // namespace oneflow
