#include "oneflow/core/operator/anchor_target_op.h"

namespace oneflow {

void AnchorTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_anchor_target_conf());

  EnrollPbInputBn("gt_boxes");

  EnrollOutputBn("rpn_labels", false);
  EnrollOutputBn("rpn_bbox_targets", false);
  EnrollOutputBn("rpn_bbox_inside_weights", false);
  EnrollOutputBn("rpn_bbox_outside_weights", false);

  EnrollConstBufBn("anchors");
  EnrollConstBufBn("inside_anchors_index");
  EnrollConstBufBn("inside_anchors_num");

  EnrollDataTmpBn("anchor_boxes_index");
  EnrollDataTmpBn("gt_boxes_absolute");
  EnrollDataTmpBn("gt_boxes_index");
  EnrollDataTmpBn("max_overlaps");
  EnrollDataTmpBn("gt_max_overlaps");
  EnrollDataTmpBn("anchor_nearest_gt_box_index");
  EnrollDataTmpBn("gt_box_nearest_anchor_index");
}

const PbMessage& AnchorTargetOp::GetCustomizedConf() const {
  return this->op_conf().anchor_target_conf();
}

const DataType AnchorTargetOp::GetDataTypeFromInputPb(const BlobDesc* gt_boxes_blob_desc) const {
  CHECK_EQ(gt_boxes_blob_desc->shape().NumAxes(), 1);
  const auto gt_boxes_type = gt_boxes_blob_desc->data_type();
  if (gt_boxes_type == DataType::kFloatList8 || gt_boxes_type == DataType::kFloatList16
      || gt_boxes_type == DataType::kFloatList24) {
    return DataType::kFloat;
  } else if (gt_boxes_type == DataType::kDoubleList8 || gt_boxes_type == DataType::kDoubleList16
             || gt_boxes_type == DataType::kDoubleList24) {
    return DataType::kDouble;
  } else {
    UNIMPLEMENTED();
  }
}

void AnchorTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const AnchorTargetOpConf& anchor_target_conf = op_conf().anchor_target_conf();
  const AnchorGeneratorConf& anchor_generator_conf = anchor_target_conf.anchor_generator_conf();
  const int32_t max_gt_boxes_num = anchor_target_conf.max_gt_boxes_num();
  const int32_t base_anchors_num =
      anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size();
  const int32_t fm_stride = anchor_generator_conf.feature_map_stride();
  CHECK_GT(fm_stride, 0);
  const int32_t fm_h = anchor_generator_conf.image_height() / fm_stride;
  const int32_t fm_w = anchor_generator_conf.image_width() / fm_stride;
  CHECK_GT(fm_h, 0);
  CHECK_GT(fm_w, 0);
  // input: gt_boxes (N) FloatList16
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  CHECK_EQ(gt_boxes_blob_desc->shape().NumAxes(), 1);
  int64_t image_num = gt_boxes_blob_desc->shape().At(0);
  const DataType bbox_data_type = GetDataTypeFromInputPb(gt_boxes_blob_desc);
  // output: rpn_labels (N, H, W, A) int32_t
  BlobDesc* rpn_labels_blob_desc = GetBlobDesc4BnInOp("rpn_labels");
  rpn_labels_blob_desc->set_data_type(DataType::kInt32);
  rpn_labels_blob_desc->mut_shape() = Shape({image_num, fm_h, fm_w, base_anchors_num});
  // output: rpn_bbox_targets (N, H, W, 4 * A) T
  BlobDesc* rpn_bbox_targets_blob_desc = GetBlobDesc4BnInOp("rpn_bbox_targets");
  rpn_bbox_targets_blob_desc->set_data_type(bbox_data_type);
  rpn_bbox_targets_blob_desc->mut_shape() = Shape({image_num, fm_h, fm_w, 4 * base_anchors_num});
  // output: rpn_bbox_inside_weights (N, H, W, 4 * A) T
  *GetBlobDesc4BnInOp("rpn_bbox_inside_weights") = *rpn_bbox_targets_blob_desc;
  // output: rpn_bbox_outside_weights (N, H, W, 4 * A) T
  *GetBlobDesc4BnInOp("rpn_bbox_outside_weights") = *rpn_bbox_targets_blob_desc;

  // const_buf: anchors (H, W, A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->set_data_type(bbox_data_type);
  anchors_blob_desc->mut_shape() = Shape({fm_h, fm_w, base_anchors_num, 4});
  // const_buf: inside_anchors_index (H * W * A) int32_t
  BlobDesc* inside_anchors_index_blob_desc = GetBlobDesc4BnInOp("inside_anchors_index");
  inside_anchors_index_blob_desc->set_data_type(DataType::kInt32);
  inside_anchors_index_blob_desc->mut_shape() = Shape({fm_h * fm_w * base_anchors_num});
  // const_buf: inside_anchors_num (1) int32_t
  BlobDesc* inside_anchors_num_blob_desc = GetBlobDesc4BnInOp("inside_anchors_num");
  inside_anchors_num_blob_desc->set_data_type(DataType::kInt32);
  inside_anchors_num_blob_desc->mut_shape() = Shape({1});

  // data_tmp: anchor_boxes_index (H * W * A) int32_t
  *GetBlobDesc4BnInOp("anchor_boxes_index") = *inside_anchors_index_blob_desc;
  // data_tmp: gt_boxes_absolute (max_gt_boxes_num, 4) T
  BlobDesc* gt_boxes_absolute_blob_desc = GetBlobDesc4BnInOp("gt_boxes_absolute");
  gt_boxes_absolute_blob_desc->set_data_type(bbox_data_type);
  gt_boxes_absolute_blob_desc->mut_shape() = Shape({max_gt_boxes_num, 4});
  // data_tmp: gt_boxes_index (max_gt_boxes_num) int32_t
  BlobDesc* gt_boxes_index_blob_desc = GetBlobDesc4BnInOp("gt_boxes_index");
  gt_boxes_index_blob_desc->set_data_type(DataType::kInt32);
  gt_boxes_index_blob_desc->mut_shape() = Shape({max_gt_boxes_num});
  // data_tmp: anchor_max_overlaps (H * W * A) float
  BlobDesc* anchor_max_overlaps_blob_desc = GetBlobDesc4BnInOp("max_overlaps");
  anchor_max_overlaps_blob_desc->set_data_type(DataType::kFloat);
  anchor_max_overlaps_blob_desc->mut_shape() = Shape({fm_h * fm_w * base_anchors_num});
  // data_tmp: anchor_nearest_gt_box_index (H * W * A) int32_t
  BlobDesc* anchor_nearest_gt_box_index_blob_desc =
      GetBlobDesc4BnInOp("anchor_nearest_gt_box_index");
  anchor_nearest_gt_box_index_blob_desc->set_data_type(DataType::kInt32);
  anchor_nearest_gt_box_index_blob_desc->mut_shape() = Shape({fm_h * fm_w * base_anchors_num});
  // data_tmp: gt_box_nearest_anchor_index (max_gt_boxes_num, H * W * A) int32_t
  BlobDesc* gt_box_nearest_anchor_index_blob_desc =
      GetBlobDesc4BnInOp("gt_box_nearest_anchor_index");
  gt_box_nearest_anchor_index_blob_desc->set_data_type(DataType::kInt32);
  gt_box_nearest_anchor_index_blob_desc->mut_shape() =
      Shape({max_gt_boxes_num, fm_h * fm_w * base_anchors_num});
  // data_tmp: gt_max_overlaps (max_gt_boxes_num) float
  BlobDesc* gt_max_overlaps_blob_desc = GetBlobDesc4BnInOp("gt_max_overlaps");
  gt_max_overlaps_blob_desc->mut_shape() = Shape({max_gt_boxes_num});
  gt_max_overlaps_blob_desc->set_data_type(DataType::kFloat);
}

void AnchorTargetOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  const auto bbox_data_type = GetDataTypeFromInputPb(gt_boxes_blob_desc);
  kernel_conf->set_data_type(bbox_data_type);
}

REGISTER_OP(OperatorConf::kAnchorTargetConf, AnchorTargetOp);

}  // namespace oneflow
