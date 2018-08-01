#include "oneflow/core/operator/bbox_nms_and_limit_op.h"

namespace oneflow {

void BboxNmsAndLimitOp::InitFromOpConf() {
  CHECK(op_conf().has_bbox_nms_and_limit_conf());
  EnrollInputBn("rois", false);
  EnrollInputBn("bbox_delta", false);
  EnrollInputBn("scores", false);
  EnrollOutputBn("labeled_bbox", false);
  EnrollOutputBn("bbox_score", false);
  EnrollDataTmpBn("bbox");
  EnrollDataTmpBn("pre_nms_index_slice");
  EnrollDataTmpBn("post_nms_index_slice");
  EnrollDataTmpBn("post_nms_keep_num");
  EnrollDataTmpBn("nms_area_tmp");
}

const PbMessage& BboxNmsAndLimitOp::GetCustomizedConf() const {
  return op_conf().bbox_nms_and_limit_conf();
}

void BboxNmsAndLimitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  // in: bbox_delta (im_num,box_num,cls_num*4) scores (im_num,box_num,cls_num) rois
  // (im_num,box_num,4)
  const BlobDesc* bbox_delta_blob_desc = GetBlobDesc4BnInOp("bbox_delta");
  const BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("scores");
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  int32_t class_num = scores_blob_desc->shape().At(2);
  CHECK_EQ(bbox_delta_blob_desc->shape().At(0), scores_blob_desc->shape().At(0));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(0), rois_blob_desc->shape().At(0));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(1), scores_blob_desc->shape().At(1));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(1), rois_blob_desc->shape().At(1));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(2), class_num * 4);
  CHECK_EQ(bbox_delta_blob_desc->shape().At(2), 4);

  // out
  BlobDesc* labeled_bbox_blob_desc = GetBlobDesc4BnInOp("labeled_bbox");
  labeled_bbox_blob_desc->mut_shape() =
      Shape({bbox_delta_blob_desc->shape().At(0)});  // todo set shape
  labeled_bbox_blob_desc->set_data_type(DataType::kOFRecord);

  BlobDesc* bbox_score_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  bbox_score_blob_desc->mut_shape() =
      Shape({bbox_delta_blob_desc->shape().At(0)});  // todo set shape
  bbox_score_blob_desc->set_data_type(DataType::kOFRecord);

  // tmp: bbox(box_num,class_num*4) pre_nms_index_slice(class_num,box_num)
  // post_nms_keep_num(class_num)
  BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  bbox_blob_desc->mut_shape() =
      Shape({bbox_delta_blob_desc->shape().At(1), bbox_delta_blob_desc->shape().At(2)});
  bbox_blob_desc->set_data_type(bbox_delta_blob_desc->data_type());

  // pre_nms_index_slice: (class_num, box_num)
  BlobDesc* pre_nms_index_slice_blob_desc = GetBlobDesc4BnInOp("pre_nms_index_slice");
  pre_nms_index_slice_blob_desc->mut_shape() =
      Shape({class_num, bbox_delta_blob_desc->shape().At(1)});
  pre_nms_index_slice_blob_desc->set_data_type(DataType::kInt32);

  *GetBlobDesc4BnInOp("post_nms_index_slice") = *pre_nms_index_slice_blob_desc;

  BlobDesc* post_nms_keep_num_blob_desc = GetBlobDesc4BnInOp("post_nms_keep_num");
  post_nms_keep_num_blob_desc->mut_shape() = Shape({scores_blob_desc->shape().At(1)});
  post_nms_keep_num_blob_desc->set_data_type(DataType::kInt32);

  // nms_area_tmp: (box_num)
  BlobDesc* nms_area_tmp_blob_desc = GetBlobDesc4BnInOp("nms_area_tmp");
  nms_area_tmp_blob_desc->mut_shape() = Shape({scores_blob_desc->shape().At(1)});
  nms_area_tmp_blob_desc->set_data_type(DataType::kInt32);

  // votting_score: (box_num, class_num)
  BlobDesc* votting_score_blob_desc = GetBlobDesc4BnInOp("votting_score");
  votting_score_blob_desc->mut_shape() =
      Shape({scores_blob_desc->shape().At(1), scores_blob_desc->shape().At(2)});
  votting_score_blob_desc->set_data_type(scores_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitOp);

}  // namespace oneflow
