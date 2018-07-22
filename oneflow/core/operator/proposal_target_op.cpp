#include "oneflow/core/operator/proposal_target_op.h"

namespace oneflow {

void ProposalTargetOp::InitFromOpConf() {
  CHECK(op_conf().has_proposal_target_conf());
  EnrollInputBn("rpn_rois", false);
  EnrollInputBn("gt_boxes", false);
  EnrollOutputBn("rois", false);
  EnrollOutputBn("labels", false);
  EnrollOutputBn("bbox_targets", false);
  EnrollOutputBn("bbox_inside_weights", false);
  EnrollOutputBn("bbox_outside_weights", false);
  EnrollDataTmpBn("all_rois");
  EnrollDataTmpBn("bbox_overlap");
  EnrollDataTmpBn("labels_tmp");
  EnrollDataTmpBn("roi_argmax");
  EnrollDataTmpBn("max_overlaps");
  EnrollDataTmpBn("fg_inds");
  EnrollDataTmpBn("bg_inds");
  EnrollDataTmpBn("fg_bg_sample_inds");
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
  CHECK_EQ(rpn_rois_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(gt_boxes_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(rpn_rois_blob_desc->shape().At(0), gt_boxes_blob_desc->shape().At(0));
  CHECK_EQ(rpn_rois_blob_desc->shape().At(2), 4);
  CHECK_EQ(gt_boxes_blob_desc->shape().At(2), 5);
  // blob shape: rpn_rois (n,roi_num,4); gt_boxes(n,gt_num,5); rois(n,roi_sample,4);
  // labels(n,roi_sample,1); bbox_target(n,roi_sample,4); bbox_inside_weights(n,roi_sample,4);
  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  BlobDesc* labels_blob_desc = GetBlobDesc4BnInOp("labels");
  BlobDesc* target_blob_desc = GetBlobDesc4BnInOp("bbox_target");
  BlobDesc* inside_weights_blob_desc = GetBlobDesc4BnInOp("bbox_inside_weights");
  BlobDesc* outside_weights_blob_desc = GetBlobDesc4BnInOp("bbox_outside_weights");
  int32_t num_roi_per_image = conf.num_roi_per_image();
  rois_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(0), num_roi_per_image, 4});
  labels_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(0), num_roi_per_image, 1});
  target_blob_desc->mut_shape() = Shape(rois_blob_desc->shape());
  inside_weights_blob_desc->mut_shape() = Shape(rois_blob_desc->shape());
  outside_weights_blob_desc->mut_shape() = Shape(rois_blob_desc->shape());

  BlobDesc* all_rois_blob_desc = GetBlobDesc4BnInOp("all_rois");
  BlobDesc* overlap_blob_desc = GetBlobDesc4BnInOp("bbox_overlap");
  BlobDesc* labels_tmp_blob_desc = GetBlobDesc4BnInOp("labels_tmp_blob");
  BlobDesc* argmax_blob_desc = GetBlobDesc4BnInOp("roi_argmax");
  BlobDesc* max_blob_desc = GetBlobDesc4BnInOp("max_overlaps");
  BlobDesc* fg_inds_blob_desc = GetBlobDesc4BnInOp("fg_inds");
  BlobDesc* bg_inds_blob_desc = GetBlobDesc4BnInOp("bg_inds");
  BlobDesc* fg_bg_sample_inds_blob_desc = GetBlobDesc4BnInOp("fg_bg_sample_inds");

  all_rois_blob_desc->mut_shape() =
      Shape({rpn_rois_blob_desc->shape().At(1) + gt_boxes_blob_desc->shape().At(1), 4});
  overlap_blob_desc->mut_shape() =
      Shape({rpn_rois_blob_desc->shape().At(1), gt_boxes_blob_desc->shape().At(1)});
  argmax_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(1), 1});
  max_blob_desc->mut_shape() = Shape(argmax_blob_desc->shape());
  fg_inds_blob_desc->mut_shape() = Shape(max_blob_desc->shape());
  bg_inds_blob_desc->mut_shape() = Shape(max_blob_desc->shape());
  fg_bg_sample_inds_blob_desc->mut_shape() = Shape({num_roi_per_image, 1});
  labels_tmp_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(1), 1});
}

REGISTER_OP(OperatorConf::kProposalTargetConf, ProposalTargetOp);
}  // namespace oneflow
