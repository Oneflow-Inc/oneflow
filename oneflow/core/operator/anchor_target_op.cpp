#include "oneflow/core/operator/anchor_target_op.h"

namespace oneflow {

void AnchorTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_anchor_target_conf());

  EnrollInputBn("gt_boxes", false);

  EnrollRepeatedOutputBn("rpn_labels", false);
  EnrollRepeatedOutputBn("rpn_bbox_targets", false);
  EnrollRepeatedOutputBn("rpn_bbox_inside_weights", false);
  EnrollRepeatedOutputBn("rpn_bbox_outside_weights", false);

  EnrollConstBufBn("anchors");
  EnrollConstBufBn("anchors_inside_inds");

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
  const AnchorTargetOpConf& op_conf = op_conf().anchor_target_conf();
  const size_t num_layers = op_conf.anchor_generator_conf_size();
  CHECK_EQ(RepeatedObnSize("rpn_labels"), num_layers);
  CHECK_EQ(RepeatedObnSize("rpn_bbox_targets"), num_layers);
  CHECK_EQ(RepeatedObnSize("rpn_bbox_inside_weights"), num_layers);
  CHECK_EQ(RepeatedObnSize("rpn_bbox_outside_weights"), num_layers);
  // input: gt_boxes (N, G, 4)
  const BlobDesc* gt_boxes_bd = GetBlobDesc4BnInOp("gt_boxes");
  int64_t num_ims = gt_boxes_bd->shape().At(0);
  int64_t num_anchors = 0;
  FOR_RANGE(size_t, layer, 0, num_layers) {
    const AnchorGeneratorConf& anchor_generator_conf : op_conf.anchor_generator_conf(layer);
    const int64_t num_anchors_per_cell =
      anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size();
    const float fm_stride = anchor_generator_conf.feature_map_stride();
    CHECK_GT(fm_stride, 0);
    const int64_t height = std::ceil(anchor_generator_conf.image_height() / fm_stride);
    const int64_t width = std::ceil(anchor_generator_conf.image_width() / fm_stride);
    CHECK_GT(height, 0);
    CHECK_GT(width, 0);
    num_anchors += height * width * num_anchors_per_cell;
    // repeat output: rpn_labels (N, H, W, A) int32_t
    BlobDesc* rpn_labels_bd = GetBlobDesc4BnInOp(RepeatedObn("rpn_labels", layer));
    rpn_labels_bd->set_data_type(DataType::kInt32);
    rpn_labels_bd->mut_shape() = Shape({num_ims, height, width, num_anchors_per_cell});
    // repeat output: rpn_bbox_targets (N, H, W, 4 * A) T
    BlobDesc* rpn_bbox_targets_bd = GetBlobDesc4BnInOp(RepeatedObn("rpn_bbox_targets", layer));
    rpn_bbox_targets_bd->set_data_type(gt_boxes_bd->data_type());
    rpn_bbox_targets_bd->mut_shape() = Shape({num_ims, height, width, num_anchors_per_cell * 4});
    // repeat output: rpn_bbox_inside_weights same as rpn_bbox_targets
    *GetBlobDesc4BnInOp(RepeatedObn("rpn_bbox_inside_weights", layer)) = *rpn_bbox_targets_bd;
    // repeat output: rpn_bbox_outside_weights same as rpn_bbox_targets
    *GetBlobDesc4BnInOp(RepeatedObn("rpn_bbox_outside_weights", layer)) = *rpn_bbox_targets_bd;
  }
  // const_buf: anchors ((H1 * W1 + H2 * W2 + ...) * A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->set_data_type(gt_boxes_bd->data_type());
  anchors_blob_desc->mut_shape() = Shape({num_anchors, 4});
  // const_buf: anchors_inside_inds (H1 * W1 + H2 * W2 + ...) * A) int32_t
  BlobDesc* anchors_inside_inds_bd = GetBlobDesc4BnInOp("anchors_inside_inds");
  anchors_inside_inds_bd->set_data_type(DataType::kInt32);
  anchors_inside_inds_bd->mut_shape() = Shape({num_anchors});
  anchors_inside_inds_bd->mut_dim0_inner_shape() = Shape({1, num_anchors});
  anchors_inside_inds_bd->set_has_dim0_valid_num_field(true);

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
