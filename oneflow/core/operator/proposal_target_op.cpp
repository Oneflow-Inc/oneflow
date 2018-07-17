#include "oneflow/core/operator/proposal_target_op.h"

namespace oneflow {

void ProposalTargetOp::InitFromOpConf() {
  CHECK(op_conf().has_proposal_target_conf());
  EnrollInputBn("rpn_rois");
  EnrollInputBn("gt_boxes");

  EnrollOutputBn("rois");
  EnrollOutputBn("labels");
  EnrollOutputBn("bbox_targets");
  EnrollOutputBn("bbox_inside_weights");
  EnrollOutputBn("bbox_outside_weights");
}

const PbMessage& ProposalTargetOp::GetCustomizedConf() const {
  return op_conf().proposal_target_conf();
}

void ProposalTargetOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const BlobDesc* rpn_rois_blob_desc = GetBlobDesc4BnInOp("rpn_rois");  //(n,roi_num,4)
  const BlobDesc* gt_boxes_blob_desc =
      GetBlobDesc4BnInOp("gt_boxes");  //(n,gt,5) n: image num; gt:gt num
  CHECK_EQ(rpn_rois_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(gt_boxes_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(rpn_rois_blob_desc->shape().At(0), gt_boxes_blob_desc->shape().At(0));
  CHECK_EQ(rpn_rois_blob_desc->shape().At(2), 4);
  CHECK_EQ(gt_boxes_blob_desc->shape().At(2), 5);

  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");           //(n,roi_sample,4);
  BlobDesc* labels_blob_desc = GetBlobDesc4BnInOp("labels");       //(n,roi_sample,1)
  BlobDesc* target_blob_desc = GetBlobDesc4BnInOp("bbox_target");  //(n,roi_sample,4);
  BlobDesc* inside_weights_blob_desc =
      GetBlobDesc4BnInOp("bbox_inside_weights");                                //(n,roi_sample,4);
  BlobDesc* outside_weights_desc = GetBlobDesc4BnInOp("bbox_outside_weights");  //(n,roi_sample,4);

  int32_t roi_batch_size = conf.roi_batch_size();
  rois_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(0), roi_batch_size, 4});
  labels_blob_desc->mut_shape() = Shape({rpn_rois_blob_desc->shape().At(0), roi_batch_size, 1});
  target_blob_desc->mut_shape() = Shape(rois_blob_desc->shape());
  inside_weights_blob_desc->mut_shape() = Shape(rois_blob_desc->shape());
  outside_weights_desc->mut_shape() = Shape(rois_blob_desc->shape());
}

REGISTER_OP(OperatorConf::kProposalTargetConf, ProposalTargetOp);
}  // namespace oneflow
