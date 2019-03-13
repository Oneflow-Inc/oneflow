#include "oneflow/core/operator/proposal_op.h"

namespace oneflow {

void ProposalOp::InitFromOpConf() {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_proposal_conf());

  EnrollInputBn("class_prob", false);
  EnrollInputBn("bbox_pred", false);
  EnrollInputBn("image_height", false);
  EnrollInputBn("image_width", false);

  EnrollOutputBn("rois", false);
  EnrollOutputBn("roi_probs", false);

  EnrollDataTmpBn("anchors");
  EnrollDataTmpBn("proposals");
  EnrollDataTmpBn("proposal_inds");
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
  // in: image_height (N) int32_t
  const BlobDesc* image_height_blob_desc = GetBlobDesc4BnInOp("image_height");
  // in: image_width (N) int32_t
  const BlobDesc* image_width_blob_desc = GetBlobDesc4BnInOp("image_width");
  CHECK_EQ(bbox_pred_blob_desc->data_type(), class_prob_blob_desc->data_type());
  CHECK_EQ(image_height_blob_desc->data_type(), DataType::kInt32);
  CHECK_EQ(image_width_blob_desc->data_type(), DataType::kInt32);
  // CHECK_EQ(bbox_pred_blob_desc->shape().NumAxes(), 4);
  // CHECK_EQ(class_prob_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(image_height_blob_desc->shape().NumAxes(), 1);
  CHECK_EQ(image_width_blob_desc->shape().NumAxes(), 1);
  const int64_t num_images = bbox_pred_blob_desc->shape().At(0);
  CHECK_EQ(num_images, class_prob_blob_desc->shape().At(0));
  CHECK_EQ(num_images, image_height_blob_desc->shape().At(0));
  CHECK_EQ(num_images, image_width_blob_desc->shape().At(0));
  const int64_t num_anchors_per_cell =
      anchor_generator_conf.aspect_ratios_size() * anchor_generator_conf.anchor_scales_size();
  CHECK_EQ(num_anchors_per_cell, class_prob_blob_desc->shape().At(3));
  CHECK_EQ(num_anchors_per_cell * 4, bbox_pred_blob_desc->shape().At(3));
  const int64_t fm_height = bbox_pred_blob_desc->shape().At(1);
  const int64_t fm_width = bbox_pred_blob_desc->shape().At(2);
  CHECK_EQ(fm_height, class_prob_blob_desc->shape().At(1));
  CHECK_EQ(fm_width, class_prob_blob_desc->shape().At(2));
  const int64_t num_anchors = fm_height * fm_width * num_anchors_per_cell;
  int64_t pre_nms_top_n = conf.pre_nms_top_n();
  if (pre_nms_top_n <= 0 || pre_nms_top_n > num_anchors) { pre_nms_top_n = num_anchors; }
  int64_t post_nms_top_n = conf.post_nms_top_n();
  if (post_nms_top_n <= 0 || post_nms_top_n > pre_nms_top_n) { post_nms_top_n = pre_nms_top_n; }

  // out: rois (num_images * post_nms_top_n, 5) T
  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  rois_blob_desc->mut_shape() = Shape({num_images * post_nms_top_n, 5});
  rois_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());
  rois_blob_desc->mut_dim0_inner_shape() = Shape({1, num_images * post_nms_top_n});
  rois_blob_desc->set_has_dim0_valid_num_field(true);
  rois_blob_desc->set_has_record_id_in_device_piece_field(true);
  // out: roi_probs (num_images * post_nms_top_n) T
  BlobDesc* roi_probs_blob_desc = GetBlobDesc4BnInOp("roi_probs");
  *roi_probs_blob_desc = *rois_blob_desc;
  roi_probs_blob_desc->mut_shape() = Shape({num_images * post_nms_top_n});

  // datatmp: anchors (H * W * A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->mut_shape() = Shape({num_anchors, 4});
  anchors_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());
  anchors_blob_desc->mut_dim0_inner_shape() = Shape({1, num_anchors});
  anchors_blob_desc->set_has_dim0_valid_num_field(true);
  // datatmp: score_slice (N, H * W * A) int32_t
  BlobDesc* proposal_inds_blob_desc = GetBlobDesc4BnInOp("proposal_inds");
  proposal_inds_blob_desc->mut_shape() = Shape({num_images, num_anchors});
  proposal_inds_blob_desc->set_data_type(DataType::kInt32);
  proposal_inds_blob_desc->set_has_dim1_valid_num_field(true);
  // datatmp: proposals (N, pre_nms_top_n, 4) T
  BlobDesc* proposal_blob_desc = GetBlobDesc4BnInOp("proposals");
  proposal_blob_desc->mut_shape() = Shape({num_images, pre_nms_top_n, 4});
  proposal_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());
  proposal_blob_desc->set_has_dim1_valid_num_field(true);
  // datatmp: pre_nms_slice (N, pre_nms_top_n) int32_t
  BlobDesc* pre_nms_slice_blob_desc = GetBlobDesc4BnInOp("pre_nms_slice");
  pre_nms_slice_blob_desc->mut_shape() = Shape({num_images, pre_nms_top_n});
  pre_nms_slice_blob_desc->set_data_type(DataType::kInt32);
  // datatmp: post_nms_slice (N, post_nms_top_n) int32_t
  BlobDesc* post_nms_slice_blob_desc = GetBlobDesc4BnInOp("post_nms_slice");
  post_nms_slice_blob_desc->mut_shape() = Shape({num_images, post_nms_top_n});
  post_nms_slice_blob_desc->set_data_type(DataType::kInt32);
  post_nms_slice_blob_desc->set_has_dim1_valid_num_field(true);
}

REGISTER_CPU_OP(OperatorConf::kProposalConf, ProposalOp);

}  // namespace oneflow
