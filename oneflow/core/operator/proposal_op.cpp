#include "oneflow/core/operator/proposal_op.h"

namespace oneflow {

void ProposalOp::InitFromOpConf() {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_proposal_conf());
  EnrollInputBn("class_prob", false);
  EnrollInputBn("bbox_pred", false);
  EnrollOutputBn("rois", false);
  EnrollOutputBn("roi_probs", false);
  EnrollConstBufBn("anchors");
  EnrollDataTmpBn("score_slice");
  EnrollDataTmpBn("proposals");
  EnrollDataTmpBn("pre_nms_slice");
  EnrollDataTmpBn("post_nms_slice");
}

const PbMessage& ProposalOp::GetCustomizedConf() const { return op_conf().proposal_conf(); }

void ProposalOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const ProposalOpConf& conf = op_conf().proposal_conf();
  const AnchorGeneratorConf& anchor_generator_conf = conf.anchor_generator_conf();
  // in: bbox_pred (N, H, W, A * 4) T
  const BlobDesc* bbox_pred_blob_desc = GetBlobDesc4BnInOp("bbox_pred");
  // in: class_prob (N, H, W, A) T
  const BlobDesc* class_prob_blob_desc = GetBlobDesc4BnInOp("class_prob");
  CHECK_EQ(bbox_pred_blob_desc->data_type(), class_prob_blob_desc->data_type());
  const int64_t num_images = bbox_pred_blob_desc->shape().At(0);
  CHECK_EQ(num_images, class_prob_blob_desc->shape().At(0));
  const int64_t num_anchors_per_cell =
      anchor_generator_conf.aspect_ratios_size() * anchor_generator_conf.anchor_scales_size();
  CHECK_EQ(num_anchors_per_cell, class_prob_blob_desc->shape().At(3));
  CHECK_EQ(num_anchors_per_cell * 4, bbox_pred_blob_desc->shape().At(3));
  const float fm_stride = anchor_generator_conf.feature_map_stride();
  const int64_t fm_height = std::ceil(anchor_generator_conf.image_height() / fm_stride);
  const int64_t fm_width = std::ceil(anchor_generator_conf.image_width() / fm_stride);
  CHECK_EQ(fm_height, bbox_pred_blob_desc->shape().At(1));
  CHECK_EQ(fm_width, bbox_pred_blob_desc->shape().At(2));
  CHECK_EQ(fm_height, class_prob_blob_desc->shape().At(1));
  CHECK_EQ(fm_width, class_prob_blob_desc->shape().At(2));
  const int64_t num_anchors = fm_height * fm_width * num_anchors_per_cell;
  int64_t pre_nms_top_n = conf.pre_nms_top_n();
  if (pre_nms_top_n <= 0 || pre_nms_top_n > num_anchors) { pre_nms_top_n = num_anchors; }
  int64_t post_nms_top_n = conf.post_nms_top_n();
  if (post_nms_top_n <= 0 || post_nms_top_n > pre_nms_top_n) { post_nms_top_n = pre_nms_top_n; }

  // const buf: anchors (H, W, A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->mut_shape() = Shape({fm_height, fm_width, num_anchors_per_cell, 4});
  anchors_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());

  // out: rois (num_images * post_nms_top_n, 5) T
  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  rois_blob_desc->mut_shape() = Shape({num_images * post_nms_top_n, 5});
  rois_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());
  rois_blob_desc->mut_dim0_inner_shape() = Shape({1, rois_blob_desc->shape().At(0)});
  rois_blob_desc->set_has_dim0_valid_num_field(true);
  rois_blob_desc->set_has_record_idx_in_device_piece_field(true);
  // out: roi_probs (num_images * post_nms_top_n) T
  BlobDesc* roi_probs_blob_desc = GetBlobDesc4BnInOp("roi_probs");
  roi_probs_blob_desc->mut_shape() = Shape({num_images * post_nms_top_n});
  roi_probs_blob_desc->set_data_type(class_prob_blob_desc->data_type());
  roi_probs_blob_desc->mut_dim0_inner_shape() = Shape({1, roi_probs_blob_desc->shape().At(0)});
  roi_probs_blob_desc->set_has_dim0_valid_num_field(true);
  roi_probs_blob_desc->set_has_record_idx_in_device_piece_field(true);

  // datatmp: score_slice (H * W * A) int32_t
  BlobDesc* score_slice_blob_desc = GetBlobDesc4BnInOp("score_slice");
  score_slice_blob_desc->mut_shape() = Shape({num_anchors});
  score_slice_blob_desc->set_data_type(DataType::kInt32);
  // datatmp: proposals (pre_nms_top_n, 4) T
  BlobDesc* proposal_blob_desc = GetBlobDesc4BnInOp("proposals");
  proposal_blob_desc->mut_shape() = Shape({pre_nms_top_n, 4});
  proposal_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());
  // datatmp: pre_nms_slice (pre_nms_top_n) int32_t
  BlobDesc* pre_nms_slice_blob_desc = GetBlobDesc4BnInOp("pre_nms_slice");
  pre_nms_slice_blob_desc->mut_shape() = Shape({pre_nms_top_n});
  pre_nms_slice_blob_desc->set_data_type(DataType::kInt32);
  // datatmp: post_nms_slice (post_nms_top_n) int32_t
  BlobDesc* post_nms_slice_blob_desc = GetBlobDesc4BnInOp("post_nms_slice");
  post_nms_slice_blob_desc->mut_shape() = Shape({post_nms_top_n});
  post_nms_slice_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kProposalConf, ProposalOp);

}  // namespace oneflow
