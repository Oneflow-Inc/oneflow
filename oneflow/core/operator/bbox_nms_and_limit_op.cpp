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
  EnrollDataTmpBn("voting_score");
  EnrollDataTmpBn("pre_nms_index_slice");
  EnrollDataTmpBn("post_nms_index_slice");
  EnrollDataTmpBn("post_nms_keep_num");
  EnrollDataTmpBn("nms_area_tmp");
}

const PbMessage& BboxNmsAndLimitOp::GetCustomizedConf() const {
  return op_conf().bbox_nms_and_limit_conf();
}

void BboxNmsAndLimitOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("bbox_delta")->data_type());
}

void BboxNmsAndLimitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  if (conf.bbox_vote_enabled()) { CHECK(conf.has_bbox_vote()); }
  // input blob shape
  // bbox_delta (n * r, c * 4)
  const BlobDesc* bbox_delta_blob_desc = GetBlobDesc4BnInOp("bbox_delta");
  // scores (n * r, c)
  const BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("scores");
  // rois (n, r, 4)
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  int64_t images_num = rois_blob_desc->shape().At(0);
  int64_t rois_num = rois_blob_desc->shape().At(1);
  int32_t class_num = scores_blob_desc->shape().At(1);
  CHECK_EQ(bbox_delta_blob_desc->shape().At(0), scores_blob_desc->shape().At(0));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(0), images_num * rois_num);
  CHECK_EQ(bbox_delta_blob_desc->shape().At(1), class_num * 4);
  CHECK_EQ(rois_blob_desc->shape().At(2), 4);
  // out blob (n) ofrecord
  BlobDesc* labeled_bbox_blob_desc = GetBlobDesc4BnInOp("labeled_bbox");
  labeled_bbox_blob_desc->mut_shape() = Shape({images_num});
  labeled_bbox_blob_desc->set_data_type(DataType::kOFRecord);
  BlobDesc* bbox_score_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  bbox_score_blob_desc->mut_shape() = Shape({images_num});
  bbox_score_blob_desc->set_data_type(DataType::kOFRecord);
  // data tmp blob shape
  // bbox (r, c, 4)
  BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  bbox_blob_desc->mut_shape() = Shape({rois_num, class_num, 4});
  bbox_blob_desc->set_data_type(bbox_delta_blob_desc->data_type());
  // voting_score (r, c)
  BlobDesc* voting_score_blob_desc = GetBlobDesc4BnInOp("voting_score");
  voting_score_blob_desc->mut_shape() = Shape({rois_num, class_num});
  voting_score_blob_desc->set_data_type(scores_blob_desc->data_type());
  // pre_nms_index_slice (c, r)
  BlobDesc* pre_nms_index_blob_desc = GetBlobDesc4BnInOp("pre_nms_index_slice");
  pre_nms_index_blob_desc->mut_shape() = Shape({class_num, rois_num});
  pre_nms_index_blob_desc->set_data_type(DataType::kInt32);
  // post_nms_index_slice (c, r)
  *GetBlobDesc4BnInOp("post_nms_index_slice") = *pre_nms_index_blob_desc;
  // post_nms_keep_num (c)
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
