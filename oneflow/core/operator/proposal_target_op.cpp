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
  EnrollOutputBn("sampled_rois", false);
  EnrollOutputBn("sampled_roi_inds", false);
  EnrollOutputBn("class_labels", false);
  EnrollOutputBn("regression_targets", false);
  EnrollOutputBn("regression_weights", false);
  // Enroll data tmp
  EnrollDataTmpBn("max_overlaps");
  EnrollDataTmpBn("max_overlaps_with_gt_index");
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

  // input: rois (R, 5) T
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  // input: gt_boxes (N, B, 4)
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  // input: gt_labels (N, B)
  const BlobDesc* gt_labels_blob_desc = GetBlobDesc4BnInOp("gt_labels");

  CHECK_EQ(rois_blob_desc->data_type(), gt_boxes_blob_desc->data_type());
  CHECK(gt_boxes_blob_desc->has_dim1_valid_num_field());
  CHECK(gt_labels_blob_desc->has_dim1_valid_num_field());
  const int64_t num_images = gt_boxes_blob_desc->shape().At(0);
  CHECK_EQ(num_images, gt_labels_blob_desc->shape().At(0));
  const int64_t num_rois = rois_blob_desc->shape().At(0);
  const int64_t total_num_sampled_rois = conf.num_sampled_rois_per_image() * num_images;
  CHECK_LE(total_num_sampled_rois, num_rois);

  // output: sampled_rois (R`, 5) T
  BlobDesc* sampled_rois_blob_desc = GetBlobDesc4BnInOp("sampled_rois");
  sampled_rois_blob_desc->mut_shape() = Shape({total_num_sampled_rois, 5});
  sampled_rois_blob_desc->set_data_type(rois_blob_desc->data_type());
  sampled_rois_blob_desc->mut_dim0_inner_shape() = Shape({1, total_num_sampled_rois});
  sampled_rois_blob_desc->set_has_dim0_valid_num_field(true);
  sampled_rois_blob_desc->set_has_record_id_in_device_piece_field(
      rois_blob_desc->has_record_id_in_device_piece_field());
  // output: sampled_roi_inds (R`) int32_t
  BlobDesc* sampled_roi_inds_blob_desc = GetBlobDesc4BnInOp("sampled_roi_inds");
  *sampled_roi_inds_blob_desc = *sampled_rois_blob_desc;
  sampled_roi_inds_blob_desc->mut_shape() = Shape({total_num_sampled_rois});
  sampled_roi_inds_blob_desc->set_data_type(DataType::kInt32);
  // output: labels (R`) int32_t
  *GetBlobDesc4BnInOp("class_labels") = *sampled_roi_inds_blob_desc;
  // output: regression_targets (R`, 4) T
  BlobDesc* regression_targets_blob_desc = GetBlobDesc4BnInOp("regression_targets");
  *regression_targets_blob_desc = *sampled_rois_blob_desc;
  regression_targets_blob_desc->mut_shape() = Shape({total_num_sampled_rois, 4});
  // output: regression_weights (R`, 4) T
  *GetBlobDesc4BnInOp("regression_weights") = *regression_targets_blob_desc;
  // TODO: Convert regression_weights to sampled_pos_inds_subset for less calculation of loss

  // data tmp: max_overlaps (R) float
  BlobDesc* max_overlaps_blob_desc = GetBlobDesc4BnInOp("max_overlaps");
  max_overlaps_blob_desc->mut_shape() = Shape({num_rois});
  max_overlaps_blob_desc->set_data_type(DataType::kFloat);
  // data tmp: max_overlaps_gt_boxes_index (R) int32_t
  BlobDesc* max_overlaps_gt_index_bd = GetBlobDesc4BnInOp("max_overlaps_with_gt_index");
  max_overlaps_gt_index_bd->mut_shape() = Shape({num_rois});
  max_overlaps_gt_index_bd->set_data_type(DataType::kInt32);
}

REGISTER_CPU_OP(OperatorConf::kProposalTargetConf, ProposalTargetOp);
}  // namespace oneflow
