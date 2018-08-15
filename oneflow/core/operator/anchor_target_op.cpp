#include "oneflow/core/operator/anchor_target_op.h"

namespace oneflow {

void AnchorTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_anchor_target_conf());

  EnrollInputBn("image_info", false);
  EnrollInputBn("gt_boxes", false);

  EnrollOutputBn("rpn_labels", false);
  EnrollOutputBn("rpn_bbox_targets", false);
  EnrollOutputBn("rpn_bbox_inside_weights", false);
  EnrollOutputBn("rpn_bbox_outside_weights", false);

  EnrollConstBufBn("anchors");
  EnrollDataTmpBn("inside_anchors_inds");
  EnrollDataTmpBn("fg_inds");
  EnrollDataTmpBn("bg_inds");
  EnrollDataTmpBn("max_overlaps");
  EnrollDataTmpBn("max_overlaps_inds");
  EnrollDataTmpBn("gt_max_overlaps_inds");
  EnrollDataTmpBn("gt_max_overlaps_num");
  EnrollDataTmpBn("origin_gt_boxes");
}

const PbMessage& AnchorTargetOp::GetCustomizedConf() const {
  return op_conf().anchor_target_conf();
}

void AnchorTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  // useful vars
  const AnchorGeneratorConf& conf = op_conf().anchor_target_conf().anchors_generator_conf();
  const int32_t base_anchors_num = conf.anchor_scales().size() * conf.aspect_ratios().size();  // A
  const int32_t fm_stride = conf.feature_map_stride();
  CHECK_GE(fm_stride, 0);
  const int32_t fm_h = conf.image_height() / fm_stride;  // H
  const int32_t fm_w = conf.image_width() / fm_stride;   // W
  CHECK_GE(fm_h, 0);
  CHECK_GE(fm_w, 0);
  const int32_t max_per_img_gt_boxes_num = 256;

  // in blobs
  // image_info: (N, 3)
  const BlobDesc* image_info_blob_desc = GetBlobDesc4BnInOp("image_info");
  CHECK_EQ(image_info_blob_desc->shape().NumAxes(), 2);
  int32_t image_num = image_info_blob_desc->shape().At(0);
  CHECK_GE(image_num, 0);
  CHECK_EQ(image_info_blob_desc->shape().At(1), 3);

  // gt_boxes: (N)
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  CHECK_EQ(gt_boxes_blob_desc->shape().NumAxes(), 1);
  CHECK_EQ(gt_boxes_blob_desc->shape().At(0), image_num);

  // out blobs
  // rpn_labels: (N, H, W, A)
  // rpn_bbox_targets: (N, H, W, 4*A)
  // rpn_bbox_inside_weights: (N, H, W, 4*A)
  // rpn_bbox_outside_weights: (N, H, W, 4*A)
  BlobDesc* rpn_labels_blob_desc = GetBlobDesc4BnInOp("rpn_labels");
  rpn_labels_blob_desc->set_data_type(DataType::kInt32);
  rpn_labels_blob_desc->mut_shape() = Shape({image_num, fm_h, fm_w, base_anchors_num});

  BlobDesc* rpn_bbox_targets_blob_desc = GetBlobDesc4BnInOp("rpn_bbox_targets");
  rpn_bbox_targets_blob_desc->set_data_type(DataType::kFloat);  // should be int
  rpn_bbox_targets_blob_desc->mut_shape() = Shape({image_num, fm_h, fm_w, 4 * base_anchors_num});

  BlobDesc* rpn_bbox_inside_weights_blob_desc = GetBlobDesc4BnInOp("rpn_bbox_inside_weights");
  rpn_bbox_inside_weights_blob_desc->set_data_type(DataType::kFloat);
  rpn_bbox_inside_weights_blob_desc->mut_shape() =
      Shape({image_num, fm_h, fm_w, 4 * base_anchors_num});

  BlobDesc* rpn_bbox_outside_weights_blob_desc = GetBlobDesc4BnInOp("rpn_bbox_outside_weights");
  rpn_bbox_outside_weights_blob_desc->set_data_type(DataType::kFloat);
  rpn_bbox_outside_weights_blob_desc->mut_shape() =
      Shape({image_num, fm_h, fm_w, 4 * base_anchors_num});

  // const blob
  // anchors: (H, W, A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");

  const auto gt_boxes_type = GetBlobDesc4BnInOp("gt_boxes")->data_type();
  if (gt_boxes_type == DataType::kFloatList8 || gt_boxes_type == DataType::kFloatList16
      || gt_boxes_type == DataType::kFloatList24) {
    anchors_blob_desc->set_data_type(DataType::kFloat);

  } else if (gt_boxes_type == DataType::kDoubleList8 || gt_boxes_type == DataType::kDoubleList16
             || gt_boxes_type == DataType::kDoubleList24) {
    anchors_blob_desc->set_data_type(DataType::kDouble);

  } else {
    UNIMPLEMENTED();
  }

  anchors_blob_desc->mut_shape() = Shape({fm_h, fm_w, base_anchors_num, 4});

  // data tmp blobs
  // inside_anchors_inds: (H*W*A) int
  // fg_inds: (H*W*A) int
  // bg_inds: (H*W*A) int
  // max_overlaps: (H*W*A) float
  // max_overlaps_inds: (H*W*A) int
  // gt_max_overlaps_inds: (256,H*W*A) int
  // gt_max_overlaps_num: (256) int
  // origin_gt_boxes: (256,4) T
  BlobDesc* inside_anchors_inds_blob_desc = GetBlobDesc4BnInOp("inside_anchors_inds");
  inside_anchors_inds_blob_desc->set_data_type(DataType::kInt32);
  inside_anchors_inds_blob_desc->mut_shape() = Shape({fm_h * fm_w * base_anchors_num});

  BlobDesc* fg_inds_blob_desc = GetBlobDesc4BnInOp("fg_inds");
  fg_inds_blob_desc->set_data_type(DataType::kInt32);
  fg_inds_blob_desc->mut_shape() = Shape({fm_h * fm_w * base_anchors_num});

  BlobDesc* bg_inds_blob_desc = GetBlobDesc4BnInOp("bg_inds");
  bg_inds_blob_desc->set_data_type(DataType::kInt32);
  bg_inds_blob_desc->mut_shape() = Shape({fm_h * fm_w * base_anchors_num});

  BlobDesc* max_overlaps_blob_desc = GetBlobDesc4BnInOp("max_overlaps");
  max_overlaps_blob_desc->set_data_type(DataType::kFloat);
  max_overlaps_blob_desc->mut_shape() = Shape({fm_h * fm_w * base_anchors_num});

  BlobDesc* max_overlaps_inds_blob_desc = GetBlobDesc4BnInOp("max_overlaps_inds");
  max_overlaps_inds_blob_desc->set_data_type(DataType::kInt32);
  max_overlaps_inds_blob_desc->mut_shape() = Shape({fm_h * fm_w * base_anchors_num});

  BlobDesc* gt_max_overlaps_inds_blob_desc = GetBlobDesc4BnInOp("gt_max_overlaps_inds");
  gt_max_overlaps_inds_blob_desc->set_data_type(DataType::kInt32);
  gt_max_overlaps_inds_blob_desc->mut_shape() =
      Shape({max_per_img_gt_boxes_num, fm_h * fm_w * base_anchors_num});

  BlobDesc* gt_max_overlaps_num_blob_desc = GetBlobDesc4BnInOp("gt_max_overlaps_num");
  gt_max_overlaps_num_blob_desc->set_data_type(DataType::kInt32);
  gt_max_overlaps_num_blob_desc->mut_shape() = Shape({max_per_img_gt_boxes_num});

  BlobDesc* gt_boxex_tmp_blob_desc = GetBlobDesc4BnInOp("origin_gt_boxes");
  if (gt_boxes_type == DataType::kFloatList8 || gt_boxes_type == DataType::kFloatList16
      || gt_boxes_type == DataType::kFloatList24) {
    gt_boxex_tmp_blob_desc->set_data_type(DataType::kFloat);

  } else if (gt_boxes_type == DataType::kDoubleList8 || gt_boxes_type == DataType::kDoubleList16
             || gt_boxes_type == DataType::kDoubleList24) {
    gt_boxex_tmp_blob_desc->set_data_type(DataType::kDouble);

  } else {
    UNIMPLEMENTED();
  }
  gt_boxex_tmp_blob_desc->mut_shape() = Shape({256, 4});
}

void AnchorTargetOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const auto gt_boxes_type = GetBlobDesc4BnInOp("gt_boxes")->data_type();
  if (gt_boxes_type == DataType::kFloatList8 || gt_boxes_type == DataType::kFloatList16
      || gt_boxes_type == DataType::kFloatList24) {
    kernel_conf->set_data_type(DataType::kFloat);
  } else if (gt_boxes_type == DataType::kDoubleList8 || gt_boxes_type == DataType::kDoubleList16
             || gt_boxes_type == DataType::kDoubleList24) {
    kernel_conf->set_data_type(DataType::kDouble);
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_OP(OperatorConf::kAnchorTargetConf, AnchorTargetOp);

}  // namespace oneflow
