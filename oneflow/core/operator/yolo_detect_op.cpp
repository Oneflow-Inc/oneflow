#include "oneflow/core/operator/operator.h"

namespace oneflow {

class YoloDetectOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(YoloDetectOp);
  YoloDetectOp() = default;
  ~YoloDetectOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_yolo_detect_conf());
    EnrollInputBn("bbox");
    EnrollInputBn("probs");
    EnrollOutputBn("out_bbox");
    EnrollOutputBn("out_probs");
    EnrollOutputBn("valid_num");
    EnrollTmpBn("select_inds");
    EnrollTmpBn("anchor_boxes_size_tmp");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().yolo_detect_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // bbox : (n, h*w*3, 4)
    // probs : (n, h*w*3, 81)
    // out_bbox : (n, h*w*3, 4)
    // out_probs : (n, h*w*3, 81)
    // select_inds : (h*w*3)
    // valid_num : (n)
    const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
    const BlobDesc* probs_blob_desc = GetBlobDesc4BnInOp("probs");
    const int32_t num_images = bbox_blob_desc->shape().At(0);
    const int32_t num_boxes = bbox_blob_desc->shape().At(1);
    int32_t max_out_boxes = num_boxes;
    if (op_conf().yolo_detect_conf().has_max_out_boxes()) {
      max_out_boxes = op_conf().yolo_detect_conf().max_out_boxes();
    }

    BlobDesc* out_bbox_blob_desc = GetBlobDesc4BnInOp("out_bbox");
    out_bbox_blob_desc->mut_shape() =
        Shape({num_images, max_out_boxes, bbox_blob_desc->shape().At(2)});
    out_bbox_blob_desc->set_data_type(bbox_blob_desc->data_type());

    BlobDesc* out_probs_blob_desc = GetBlobDesc4BnInOp("out_probs");
    out_probs_blob_desc->mut_shape() =
        Shape({num_images, max_out_boxes, probs_blob_desc->shape().At(2)});
    out_probs_blob_desc->set_data_type(probs_blob_desc->data_type());

    // data_tmp: select_inds (h*w*3) kInt32
    BlobDesc* valid_num_blob_desc = GetBlobDesc4BnInOp("valid_num");
    valid_num_blob_desc->set_data_type(DataType::kInt32);
    valid_num_blob_desc->mut_shape() = Shape({num_images});

    // data_tmp: select_inds (h*w*3) kInt32
    BlobDesc* select_inds_blob_desc = GetBlobDesc4BnInOp("select_inds");
    select_inds_blob_desc->set_data_type(DataType::kInt32);
    select_inds_blob_desc->mut_shape() = Shape({num_boxes});

    // data_tmp: anchor_boxes_size_tmp (conf.anchor_boxes_size_size()*2) int32_t
    BlobDesc* anchor_boxes_size_tmp_blob_desc = GetBlobDesc4BnInOp("anchor_boxes_size_tmp");
    anchor_boxes_size_tmp_blob_desc->set_data_type(DataType::kInt32);
    anchor_boxes_size_tmp_blob_desc->mut_shape() =
        Shape({op_conf().yolo_detect_conf().anchor_boxes_size_size() * 2});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("bbox", 0)
        .Split("probs", 0)
        .Split("out_bbox", 0)
        .Split("out_probs", 0)
        .Split("valid_num", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());

    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("valid_num") = *BatchAxis4BnInOp("bbox");
    *BatchAxis4BnInOp("out_bbox") = *BatchAxis4BnInOp("bbox");
    *BatchAxis4BnInOp("out_probs") = *BatchAxis4BnInOp("bbox");
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kYoloDetectConf, YoloDetectOp);

}  // namespace oneflow
