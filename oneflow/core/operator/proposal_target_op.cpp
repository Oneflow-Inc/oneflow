#include "oneflow/core/operator/proposal_target_op.h"

namespace oneflow {

void ProposalTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_proposal_target_conf());
  // Enroll input
  EnrollInputBn("rpn_rois", false);
  EnrollPbInputBn("gt_boxes");
  EnrollPbInputBn("gt_labels");
  // Enroll output
  EnrollOutputBn("rois", false);
  EnrollOutputBn("labels", false);
  EnrollOutputBn("bbox_targets", false);
  EnrollOutputBn("bbox_inside_weights", false);
  EnrollOutputBn("bbox_outside_weights", false);
  // Enroll data tmp
  EnrollDataTmpBn("boxes_index");
  EnrollDataTmpBn("max_overlaps");
  EnrollDataTmpBn("max_overlaps_gt_boxes_index");
}

const PbMessage& ProposalTargetOp::GetCustomizedConf() const {
  return op_conf().proposal_target_conf();
}

void ProposalTargetOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  // TODO: Check conf
  // input: rpn_rois (n, r, 4) T
  const BlobDesc* rpn_rois_blob_desc = GetBlobDesc4BnInOp("rpn_rois");
  // input: gt_boxes (n) FloatList16 (r * 4)
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  // input: gt_labels (n) Int32List16 (r)
  const BlobDesc* gt_labels_blob_desc = GetBlobDesc4BnInOp("gt_labels");
  int64_t image_num = rpn_rois_blob_desc->shape().At(0);
  CHECK_EQ(image_num, gt_boxes_blob_desc->shape().At(0));
  CHECK_EQ(image_num, gt_labels_blob_desc->shape().At(0));
  int64_t rois_num = rpn_rois_blob_desc->shape().At(1);
  int64_t output_num = conf.num_rois_per_image();
  int64_t class_num = conf.num_classes();
  int64_t max_gt_boxes_num = conf.max_gt_boxes_num();
  DataType data_type = rpn_rois_blob_desc->data_type();
  // output: rois (n, output_num, 4) T
  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  rois_blob_desc->mut_shape() = Shape({image_num, output_num, 4});
  rois_blob_desc->set_data_type(data_type);
  rois_blob_desc->set_has_data_id_field(rpn_rois_blob_desc->has_data_id_field());
  // output: labels (n * output_num) int32_t
  BlobDesc* labels_blob_desc = GetBlobDesc4BnInOp("labels");
  labels_blob_desc->mut_shape() = Shape({image_num * output_num});
  labels_blob_desc->set_data_type(DataType::kInt32);
  // output: bbox_targets (n * output_num, class_num * 4) T
  BlobDesc* bbox_targets_blob_desc = GetBlobDesc4BnInOp("bbox_targets");
  bbox_targets_blob_desc->mut_shape() = Shape({image_num * output_num, class_num * 4});
  bbox_targets_blob_desc->set_data_type(data_type);
  // output: bbox_targets (n * output_num, class_num * 4) T
  *GetBlobDesc4BnInOp("bbox_inside_weights") = *bbox_targets_blob_desc;
  // output: bbox_targets (n * output_num, class_num * 4) T
  *GetBlobDesc4BnInOp("bbox_outside_weights") = *bbox_targets_blob_desc;
  // data tmp: boxes_index (rois_num + max_gt_boxes_num) int32_t
  BlobDesc* boxes_index_blob_desc = GetBlobDesc4BnInOp("boxes_index");
  boxes_index_blob_desc->mut_shape() = Shape({rois_num + max_gt_boxes_num});
  boxes_index_blob_desc->set_data_type(DataType::kInt32);
  // data tmp: max_overlaps (rois_num + max_gt_boxes_num) float
  BlobDesc* max_overlaps_blob_desc = GetBlobDesc4BnInOp("max_overlaps");
  max_overlaps_blob_desc->mut_shape() = Shape({rois_num + max_gt_boxes_num});
  max_overlaps_blob_desc->set_data_type(DataType::kFloat);
  // data tmp: max_overlaps_gt_boxes_index (rois_num + max_gt_boxes_num) int32_t
  BlobDesc* max_overlaps_gt_boxes_index_blob_desc =
      GetBlobDesc4BnInOp("max_overlaps_gt_boxes_index");
  max_overlaps_gt_boxes_index_blob_desc->mut_shape() = Shape({rois_num + max_gt_boxes_num});
  max_overlaps_gt_boxes_index_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kProposalTargetConf, ProposalTargetOp);
}  // namespace oneflow
