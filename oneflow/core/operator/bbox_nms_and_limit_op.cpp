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
  // input: bbox_delta (n * r, c * 4)
  const BlobDesc* bbox_delta_blob_desc = GetBlobDesc4BnInOp("bbox_delta");
  // input: scores (n * r, c)
  const BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("scores");
  // input: rois (n, r, 4)
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  const int64_t images_num = rois_blob_desc->shape().At(0);
  const int64_t rois_num = rois_blob_desc->shape().At(1);
  const int64_t class_num = scores_blob_desc->shape().At(1);
  CHECK_EQ(bbox_delta_blob_desc->shape().At(0), scores_blob_desc->shape().At(0));
  CHECK_EQ(bbox_delta_blob_desc->shape().At(0), images_num * rois_num);
  CHECK_EQ(bbox_delta_blob_desc->shape().At(1), class_num * 4);
  CHECK_EQ(rois_blob_desc->shape().At(2), 4);
  // output: labeled_bbox (n) pb
  BlobDesc* labeled_bbox_blob_desc = GetBlobDesc4BnInOp("labeled_bbox");
  labeled_bbox_blob_desc->mut_shape() = Shape({images_num});
  labeled_bbox_blob_desc->set_data_type(DataType::kInt32List16);
  labeled_bbox_blob_desc->set_has_data_id_field(rois_blob_desc->has_data_id_field());
  // output: bbox_score (n) pb
  BlobDesc* bbox_score_blob_desc = GetBlobDesc4BnInOp("bbox_score");
  bbox_score_blob_desc->mut_shape() = Shape({images_num});
  bbox_score_blob_desc->set_data_type(DataType::kFloatList16);
  bbox_score_blob_desc->set_has_data_id_field(rois_blob_desc->has_data_id_field());
  // datatmp: bbox (r, c, 4)
  BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  bbox_blob_desc->mut_shape() = Shape({rois_num, class_num, 4});
  bbox_blob_desc->set_data_type(bbox_delta_blob_desc->data_type());
  // datatmp: voting_score (r, c)
  BlobDesc* voting_score_blob_desc = GetBlobDesc4BnInOp("voting_score");
  voting_score_blob_desc->mut_shape() = Shape({rois_num, class_num});
  voting_score_blob_desc->set_data_type(scores_blob_desc->data_type());
  // datatmp: pre_nms_index_slice (c, r)
  BlobDesc* pre_nms_index_blob_desc = GetBlobDesc4BnInOp("pre_nms_index_slice");
  pre_nms_index_blob_desc->mut_shape() = Shape({class_num, rois_num});
  pre_nms_index_blob_desc->set_data_type(DataType::kInt32);
  // datatmp: post_nms_index_slice (c, r)
  *GetBlobDesc4BnInOp("post_nms_index_slice") = *pre_nms_index_blob_desc;
}

REGISTER_OP(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitOp);

}  // namespace oneflow
