#include "oneflow/core/operator/anchor_target_op.h"

namespace oneflow {

void AnchorTargetOp::InitFromOpConf() {
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
  EnrollDataTmpBn("inds_mask");
}

const PbMessage& AnchorTargetOp::GetCustomizedConf() const {
  return op_conf().anchor_target_conf();
}

void AnchorTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  // useful vars
  const AnchorGeneratorConf& conf = op_conf().anchor_target_conf().anchors_generator_conf();
  const int32_t per_pixel_anchors_num =
      conf.anchor_scales().size() * conf.aspect_ratios().size();  // A
  const int32_t fm_stride = conf.feature_map_stride();
  CHECK_GE(fm_stride, 0);
  const int32_t fm_h = conf.image_height() / fm_stride;  // H
  const int32_t fm_w = conf.image_width() / fm_stride;   // W
  CHECK_GE(fm_h, 0);
  CHECK_GE(fm_w, 0);

  // in blobs
  // image_info: (N, 3)
  const BlobDesc* image_info_blob_desc = GetBlobDesc4BnInOp("image_info");
  CHECK_EQ(image_info_blob_desc->shape().NumAxes(), 2);
  int32_t image_num = image_info_blob_desc->shape().At(0);
  CHECK_GE(image_num, 0);
  CHECK_EQ(image_info_blob_desc->shape().At(1), 3);

  // gt_boxes: (N, 256, 4)
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  // CHECK_EQ(gt_boxes_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(gt_boxes_blob_desc->shape().At(0), image_num);
  const int32_t max_per_img_gt_boxes_num = 256;  // gt_boxes_blob_desc->shape().At(1);  // 256
  // CHECK_EQ(gt_boxes_blob_desc->shape().At(2), 4);

  // out blobs
  // rpn_labels: (N, H, W, A)
  // rpn_bbox_targets: (N, H, W, 4*A)
  // rpn_bbox_inside_weights: (N, H, W, 4*A)
  // rpn_bbox_outside_weights: (N, H, W, 4*A)
  BlobDesc* rpn_labels_blob_desc = GetBlobDesc4BnInOp("rpn_labels");
  rpn_labels_blob_desc->set_data_type(DataType::kInt32);
  rpn_labels_blob_desc->mut_shape() = Shape({image_num, fm_h, fm_w, per_pixel_anchors_num});

  BlobDesc* rpn_bbox_targets_blob_desc = GetBlobDesc4BnInOp("rpn_bbox_targets");
  rpn_bbox_targets_blob_desc->set_data_type(DataType::kFloat);  // should be int
  rpn_bbox_targets_blob_desc->mut_shape() =
      Shape({image_num, fm_h, fm_w, 4 * per_pixel_anchors_num});

  BlobDesc* rpn_bbox_inside_weights_blob_desc = GetBlobDesc4BnInOp("rpn_bbox_inside_weights");
  rpn_bbox_inside_weights_blob_desc->set_data_type(DataType::kFloat);
  rpn_bbox_inside_weights_blob_desc->mut_shape() =
      Shape({image_num, fm_h, fm_w, 4 * per_pixel_anchors_num});

  BlobDesc* rpn_bbox_outside_weights_blob_desc = GetBlobDesc4BnInOp("rpn_bbox_outside_weights");
  rpn_bbox_outside_weights_blob_desc->set_data_type(DataType::kFloat);
  rpn_bbox_outside_weights_blob_desc->mut_shape() =
      Shape({image_num, fm_h, fm_w, 4 * per_pixel_anchors_num});

  // const blob
  // anchors: (H, W, 4*A)
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->set_data_type(gt_boxes_blob_desc->data_type());
  anchors_blob_desc->mut_shape() = Shape({fm_h, fm_w, 4 * per_pixel_anchors_num});

  // data tmp blobs
  // inside_anchors_inds: (H*W*A)
  // fg_inds: (H*W*A)
  // bg_inds: (H*W*A)
  // max_overlaps: (H*W*A)
  // max_overlaps_inds: (H*W*A)
  // gt_max_overlaps_inds: (256,H*W*A)
  // gt_max_overlaps_num: (256)
  // inds_mask: (H*W*A)
  BlobDesc* inside_anchors_inds_blob_desc = GetBlobDesc4BnInOp("inside_anchors_inds");
  inside_anchors_inds_blob_desc->set_data_type(DataType::kInt32);
  inside_anchors_inds_blob_desc->mut_shape() = Shape({fm_h * fm_w * per_pixel_anchors_num});

  BlobDesc* fg_inds_blob_desc = GetBlobDesc4BnInOp("fg_inds");
  fg_inds_blob_desc->set_data_type(DataType::kInt32);
  fg_inds_blob_desc->mut_shape() = Shape({fm_h * fm_w * per_pixel_anchors_num});

  BlobDesc* bg_inds_blob_desc = GetBlobDesc4BnInOp("bg_inds");
  bg_inds_blob_desc->set_data_type(DataType::kInt32);
  bg_inds_blob_desc->mut_shape() = Shape({fm_h * fm_w * per_pixel_anchors_num});

  BlobDesc* max_overlaps_blob_desc = GetBlobDesc4BnInOp("max_overlaps");
  max_overlaps_blob_desc->set_data_type(DataType::kFloat);
  max_overlaps_blob_desc->mut_shape() = Shape({fm_h * fm_w * per_pixel_anchors_num});

  BlobDesc* max_overlaps_inds_blob_desc = GetBlobDesc4BnInOp("max_overlaps_inds");
  max_overlaps_inds_blob_desc->set_data_type(DataType::kInt32);
  max_overlaps_inds_blob_desc->mut_shape() = Shape({fm_h * fm_w * per_pixel_anchors_num});

  BlobDesc* gt_max_overlaps_inds_blob_desc = GetBlobDesc4BnInOp("gt_max_overlaps_inds");
  gt_max_overlaps_inds_blob_desc->set_data_type(DataType::kInt32);
  gt_max_overlaps_inds_blob_desc->mut_shape() =
      Shape({max_per_img_gt_boxes_num, fm_h * fm_w * per_pixel_anchors_num});

  BlobDesc* gt_max_overlaps_num_blob_desc = GetBlobDesc4BnInOp("gt_max_overlaps_num");
  gt_max_overlaps_num_blob_desc->set_data_type(DataType::kInt32);
  gt_max_overlaps_inds_blob_desc->mut_shape() = Shape({max_per_img_gt_boxes_num});

  BlobDesc* inds_mask_blob_desc = GetBlobDesc4BnInOp("inds_mask");
  inds_mask_blob_desc->set_data_type(DataType::kInt32);
  inds_mask_blob_desc->mut_shape() = Shape({fm_h * fm_w * per_pixel_anchors_num});
}

REGISTER_OP(OperatorConf::kAnchorTargetConf, AnchorTargetOp);

}  // namespace oneflow
