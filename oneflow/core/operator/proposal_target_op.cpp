#include "oneflow/core/operator/proposal_target_op.h"

namespace oneflow {

void ProposalTargetOp::InitFromOpConf() {
  CHECK(op_conf().has_proposal_target_conf());
  EnrollInputBn("rpn_rois", false);
  EnrollInputBn("gt_boxes", false);
  EnrollInputBn("gt_label", false);
  EnrollInputBn("im_info", false);
  EnrollOutputBn("rois", false);
  EnrollOutputBn("labels", false);
  EnrollOutputBn("bbox_targets", false);
  EnrollOutputBn("bbox_inside_weights", false);
  EnrollOutputBn("bbox_outside_weights", false);
  EnrollDataTmpBn("roi_nearest_gt_index");
  EnrollDataTmpBn("roi_max_overlap");
  EnrollDataTmpBn("rois_index");
}

const PbMessage& ProposalTargetOp::GetCustomizedConf() const {
  return op_conf().proposal_target_conf();
}

void ProposalTargetOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const BlobDesc* rpn_rois_blob_desc = GetBlobDesc4BnInOp("rpn_rois");
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  const BlobDesc* im_info_blob_desc = GetBlobDesc4BnInOp("im_info");
  const BlobDesc* gt_label_blob_desc = GetBlobDesc4BnInOp("gt_label");
  CHECK_EQ(rpn_rois_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(gt_boxes_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(rpn_rois_blob_desc->shape().At(0), gt_boxes_blob_desc->shape().At(0));
  CHECK_EQ(im_info_blob_desc->shape().At(0), gt_boxes_blob_desc->shape().At(0));
  CHECK_EQ(rpn_rois_blob_desc->shape().At(2), gt_boxes_blob_desc->shape().At(2));
  CHECK_EQ(gt_label_blob_desc->shape().At(1), gt_boxes_blob_desc->shape().At(1));
  // blob shape: rpn_rois (n,roi_num,4); gt_boxes(n,gt_max_num,4); im_info(n,3);
  // rois(n,roi_sample,4); labels(n*roi_sample); bbox_target(n*roi_sample,4*class);
  // bbox_inside_weights(n,roi_sample,4*class); bbox_outside_weights (n,roi_sample,4*class);
  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  BlobDesc* labels_blob_desc = GetBlobDesc4BnInOp("labels");
  BlobDesc* target_blob_desc = GetBlobDesc4BnInOp("bbox_target");
  BlobDesc* inside_weights_blob_desc = GetBlobDesc4BnInOp("bbox_inside_weights");
  BlobDesc* outside_weights_blob_desc = GetBlobDesc4BnInOp("bbox_outside_weights");
  int64_t num_roi_per_image = conf.num_roi_per_image();
  int64_t class_num = conf.class_num();
  rois_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(0), num_roi_per_image, 4});
  labels_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(0) * num_roi_per_image});
  target_blob_desc->mut_shape() =
      Shape({rpn_rois_blob_desc->shape().At(0) * num_roi_per_image, 4 * class_num});
  inside_weights_blob_desc->mut_shape() = Shape(target_blob_desc->shape());
  outside_weights_blob_desc->mut_shape() = Shape(target_blob_desc->shape());
  // tmp blob shape: roi_nearest_gt_index (roi_num); roi_max_overlap(roi_num) rois_index(roi_num)
  BlobDesc* roi_nearest_gt_index_blob_desc = GetBlobDesc4BnInOp("roi_nearest_gt_index");
  BlobDesc* roi_max_overlap_blob_desc = GetBlobDesc4BnInOp("roi_max_overlap");
  BlobDesc* rois_index_blob_desc = GetBlobDesc4BnInOp("rois_index");

  roi_nearest_gt_index_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(1)});
  roi_max_overlap_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(1)});
  rois_index_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(1)});
}

REGISTER_OP(OperatorConf::kProposalTargetConf, ProposalTargetOp);
}  // namespace oneflow
