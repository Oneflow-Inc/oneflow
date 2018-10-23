#include "oneflow/core/operator/proposal_target_op.h"

namespace oneflow {

void ProposalTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_proposal_target_conf());
  // Enroll input
  EnrollInputBn("rois", false);
  EnrollInputBn("gt_boxes", false);
  EnrollInputBn("gt_labels", false);
  // Enroll output
  EnrollOutputBn("out_rois", false);
  EnrollOutputBn("labels", false);
  EnrollOutputBn("bbox_targets", false);
  EnrollOutputBn("bbox_inside_weights", false);
  EnrollOutputBn("bbox_outside_weights", false);
  // Enroll data tmp
  EnrollDataTmpBn("gt_boxes_inds");
  EnrollDataTmpBn("rois_inds");
  EnrollDataTmpBn("max_overlaps");
  EnrollDataTmpBn("max_overlaps_with_gt_index");
  EnrollDataTmpBn("sampled_inds");
}

const PbMessage& ProposalTargetOp::GetCustomizedConf() const {
  return op_conf().proposal_target_conf();
}

void ProposalTargetOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  CHECK_GE(conf.foreground_threshold(), conf.background_threshold_high());
  CHECK_GE(conf.background_threshold_high(), conf.background_threshold_low());
  CHECK_GE(conf.background_threshold_low(), 0.f);
  // input: rois (r, 5) T
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  // input: gt_boxes (n, g, 4)
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  // input: gt_labels (n, g)
  const BlobDesc* gt_labels_blob_desc = GetBlobDesc4BnInOp("gt_labels");
  CHECK_EQ(rois_blob_desc->data_type(), gt_boxes_blob_desc->data_type());
  CHECK(gt_boxes_blob_desc->has_dim1_valid_num_field());
  CHECK(gt_labels_blob_desc->has_dim1_valid_num_field());
  const int64_t num_images = gt_boxes_blob_desc->shape().At(0);
  const int64_t num_gt = gt_boxes_blob_desc->shape().Count(0, 2);
  CHECK_EQ(gt_labels_blob_desc->shape().At(0), num_images);
  const int64_t num_rois = rois_blob_desc->shape().At(0);
  const int64_t total_num_sampled_rois = conf.num_sampled_rois_per_image() * num_images;
  CHECK_LE(total_num_sampled_rois, num_rois);
  const int64_t num_classes = conf.num_classes();

  // output: rois (total_num_sampled_rois, 5) T
  BlobDesc* out_rois_blob_desc = GetBlobDesc4BnInOp("out_rois");
  out_rois_blob_desc->mut_shape() = Shape({total_num_sampled_rois, 5});
  out_rois_blob_desc->set_data_type(rois_blob_desc->data_type());
  out_rois_blob_desc->mut_dim0_inner_shape() = Shape({1, total_num_sampled_rois});
  out_rois_blob_desc->set_has_dim0_valid_num_field(true);
  out_rois_blob_desc->set_has_record_idx_in_device_piece_field(
      rois_blob_desc->has_record_idx_in_device_piece_field());
  // output: labels (total_num_sampled_rois) int32_t
  BlobDesc* labels_blob_desc = GetBlobDesc4BnInOp("labels");
  labels_blob_desc->mut_shape() = Shape({total_num_sampled_rois});
  labels_blob_desc->set_data_type(DataType::kInt32);
  labels_blob_desc->mut_dim0_inner_shape() = Shape({1, total_num_sampled_rois});
  labels_blob_desc->set_has_dim0_valid_num_field(true);
  labels_blob_desc->set_has_record_idx_in_device_piece_field(
      rois_blob_desc->has_record_idx_in_device_piece_field());
  // output: bbox_targets (total_num_sampled_rois, num_classes * 4) T
  BlobDesc* bbox_targets_blob_desc = GetBlobDesc4BnInOp("bbox_targets");
  bbox_targets_blob_desc->mut_shape() = Shape({total_num_sampled_rois, num_classes * 4});
  bbox_targets_blob_desc->set_data_type(rois_blob_desc->data_type());
  bbox_targets_blob_desc->mut_dim0_inner_shape() = Shape({1, total_num_sampled_rois});
  bbox_targets_blob_desc->set_has_dim0_valid_num_field(true);
  bbox_targets_blob_desc->set_has_record_idx_in_device_piece_field(
      rois_blob_desc->has_record_idx_in_device_piece_field());
  // output: bbox_inside_weights (total_num_sampled_rois, num_classes * 4) T
  *GetBlobDesc4BnInOp("bbox_inside_weights") = *bbox_targets_blob_desc;
  // output: bbox_outside_weights (total_num_sampled_rois, num_classes * 4) T
  *GetBlobDesc4BnInOp("bbox_outside_weights") = *bbox_targets_blob_desc;

  // data tmp: gt_boxes_inds (n * g) int32_t
  BlobDesc* gt_boxes_inds_blob_desc = GetBlobDesc4BnInOp("gt_boxes_inds");
  gt_boxes_inds_blob_desc->mut_shape() = Shape({num_gt});
  gt_boxes_inds_blob_desc->set_data_type(DataType::kInt32);
  // data tmp: rois_inds (r + n * g) int32_t
  BlobDesc* rois_inds_blob_desc = GetBlobDesc4BnInOp("rois_inds");
  rois_inds_blob_desc->mut_shape() = Shape({num_rois + num_gt});
  rois_inds_blob_desc->set_data_type(DataType::kInt32);
  // data tmp: max_overlaps (r + n * g) float
  BlobDesc* max_overlaps_blob_desc = GetBlobDesc4BnInOp("max_overlaps");
  max_overlaps_blob_desc->mut_shape() = Shape({num_rois + num_gt});
  max_overlaps_blob_desc->set_data_type(DataType::kFloat);
  // data tmp: max_overlaps_gt_boxes_index (r + n * g) int32_t
  BlobDesc* max_overlaps_gt_index_bd = GetBlobDesc4BnInOp("max_overlaps_with_gt_index");
  max_overlaps_gt_index_bd->mut_shape() = Shape({num_rois + num_gt});
  max_overlaps_gt_index_bd->set_data_type(DataType::kInt32);
  // data tmp: sampled_inds (total_num_sampled_rois) int32_t
  BlobDesc* sampled_inds_blob_desc = GetBlobDesc4BnInOp("sampled_inds");
  sampled_inds_blob_desc->mut_shape() = Shape({total_num_sampled_rois});
  sampled_inds_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kProposalTargetConf, ProposalTargetOp);
}  // namespace oneflow
