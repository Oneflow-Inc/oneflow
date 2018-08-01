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
  // input blob shape
  // bbox_delta (n, r, c * 4)
  // scores (n, r, c)
  // rois (n, r, 4)
  const BlobDesc* bbox_delta_blob_desc = GetBlobDesc4BnInOp("bbox_delta");
  const BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("scores");
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  int32_t class_num = scores_blob_desc->shape().At(2);
  CHECK_EQ(bbox_delta_blob_desc->shape().At(0), scores_blob_desc->shape().At(0));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(0), rois_blob_desc->shape().At(0));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(1), scores_blob_desc->shape().At(1));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(1), rois_blob_desc->shape().At(1));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(2), class_num * 4);
  CHECK_EQ(rois_blob_desc->shape().At(2), 4);
  int64_t images_num = bbox_delta_blob_desc->shape().At(0);
  int64_t rois_num = rois_blob_desc->shape().At(1);
  // out blob
  BlobDesc* labeled_bbox_blob_desc = GetBlobDesc4BnInOp("labeled_bbox");
  labeled_bbox_blob_desc->mut_shape() = Shape({images_num});
  labeled_bbox_blob_desc->set_data_type(DataType::kOFRecord);
  BlobDesc* bbox_score_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  bbox_score_blob_desc->mut_shape() = Shape({images_num});
  bbox_score_blob_desc->set_data_type(DataType::kOFRecord);
  // data tmp blob shape
  // bbox (r, c * 4)
  // votting_score (r, c)
  BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  bbox_blob_desc->mut_shape() = Shape({rois_num, class_num * 4});
  bbox_blob_desc->set_data_type(bbox_delta_blob_desc->data_type());
  BlobDesc* votting_score_blob_desc = GetBlobDesc4BnInOp("votting_score");
  votting_score_blob_desc->mut_shape() = Shape({rois_num, class_num});
  votting_score_blob_desc->set_data_type(scores_blob_desc->data_type());
  // pre_nms_index_slice (c, r)
  // post_nms_index_slice (c, r)
  // post_nms_keep_num (c)
  BlobDesc* pre_nms_index_blob_desc = GetBlobDesc4BnInOp("pre_nms_index_slice");
  pre_nms_index_blob_desc->mut_shape() = Shape({class_num, rois_num});
  pre_nms_index_blob_desc->set_data_type(DataType::kInt32);
  *GetBlobDesc4BnInOp("post_nms_index_slice") = *pre_nms_index_blob_desc;
  BlobDesc* post_nms_keep_num_blob_desc = GetBlobDesc4BnInOp("post_nms_keep_num");
  post_nms_keep_num_blob_desc->mut_shape() = Shape({class_num});
  post_nms_keep_num_blob_desc->set_data_type(DataType::kInt32);
  // nms_area_tmp (r)
  BlobDesc* nms_area_tmp_blob_desc = GetBlobDesc4BnInOp("nms_area_tmp");
  nms_area_tmp_blob_desc->mut_shape() = Shape({rois_num});
  nms_area_tmp_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitOp);

}  // namespace oneflow
