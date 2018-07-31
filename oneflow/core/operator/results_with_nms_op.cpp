#include "oneflow/core/operator/results_with_nms_op.h"

namespace oneflow {

void ResultsWithNmsOp::InitFromOpConf() {
  CHECK(op_conf().has_results_with_nms_conf());
  EnrollInputBn("boxes", false);
  EnrollInputBn("scores", false);
  EnrollOutputBn("cls_boxes", false);
  EnrollOutputBn("out_boxes", false);
  EnrollOutputBn("out_scores", false);
  EnrollDataTmpBn("sorted_score_index");
  EnrollDataTmpBn("suppressed_index");
  EnrollDataTmpBn("score_per_class");
  EnrollDataTmpBn("box_per_class");
  EnrollDataTmpBn("nms_boxes");
  EnrollDataTmpBn("nms_scores");
}

const PbMessage& ResultsWithNmsOp::GetCustomizedConf() const {
  return op_conf().results_with_nms_conf();
}

void ResultsWithNmsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  const ResultsWithNmsOpConf& conf = op_conf().results_with_nms_conf();
  int32_t class_num = conf.class_num();
  // in: boxes (im_num,box_num,4*cls_num) scores (im_num,box_num,1*cls_num)
  const BlobDesc* boxes_blob_desc = GetBlobDesc4BnInOp("boxes");
  const BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("scores");
  CHECK_EQ(boxes_blob_desc->shape().At(0), scores_blob_desc->shape().At(0));
  CHECK_EQ(boxes_blob_desc->shape().At(1), scores_blob_desc->shape().At(1));
  CHECK_EQ(boxes_blob_desc->shape().At(2), class_num * 4);
  CHECK_EQ(scores_blob_desc->shape().At(2), class_num);
  // out: cls_boxes (im_num,class_num,box_num*5) out_boxes(im_num,box_num*cls_num,4)
  // out_scores(im_num,box_num*cls_num,1)
  BlobDesc* cls_boxes_blob_desc = GetBlobDesc4BnInOp("cls_boxes");
  cls_boxes_blob_desc->mut_shape() =
      Shape({boxes_blob_desc->shape().At(0), class_num, scores_blob_desc->shape().At(1) * 5});

  BlobDesc* out_boxes_blob_desc = GetBlobDesc4BnInOp("out_boxes");
  out_boxes_blob_desc->mut_shape() =
      Shape({boxes_blob_desc->shape().At(0), scores_blob_desc->shape().At(1) * class_num, 4});

  BlobDesc* out_scores_blob_desc = GetBlobDesc4BnInOp("out_scores");
  out_scores_blob_desc->mut_shape() =
      Shape({boxes_blob_desc->shape().At(0), scores_blob_desc->shape().At(1) * class_num});
  
  // sorted score slice in nms section: (box_num) 
  // sorted score slice in limiting section: (class_num*box_num)
  BlobDesc* sorted_score_index_blob_desc = GetBlobDesc4BnInOp("sorted_score_sclice");
  sorted_score_index_blob_desc->mut_shape() =
      Shape({class_num * boxes_blob_desc->shape().At(1)});
  sorted_score_index_blob_desc->set_data_type(DataType::kInt32);

  BlobDesc* box_per_class_blob_desc = GetBlobDesc4BnInOp("box_per_class");
  box_per_class_blob_desc->mut_shape() = Shape({boxes_blob_desc->shape().At(1), 4});
  box_per_class_blob_desc->set_data_type(boxes_blob_desc->data_type());

  BlobDesc* score_per_class_blob_desc = GetBlobDesc4BnInOp("score_per_class");
  score_per_class_blob_desc->mut_shape() = Shape({boxes_blob_desc->shape().At(1)});
  score_per_class_blob_desc->set_data_type(scores_blob_desc->data_type());

  BlobDesc* nms_boxes_blob_desc = GetBlobDesc4BnInOp("post_nms_boxes");
  nms_boxes_blob_desc->mut_shape() = Shape({class_num, boxes_blob_desc->shape().At(1), 4});
  nms_boxes_blob_desc->set_data_type(boxes_blob_desc->data_type());

  BlobDesc* nms_scores_blob_desc = GetBlobDesc4BnInOp("post_nms_scores");
  nms_scores_blob_desc->mut_shape() = Shape({class_num, boxes_blob_desc->shape().At(1)});
  nms_scores_blob_desc->set_data_type(scores_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kResultsWithNmsConf, ResultsWithNmsOp);

}  // namespace oneflow
