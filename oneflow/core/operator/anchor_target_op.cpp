#include "oneflow/core/operator/anchor_target_op.h"

namespace oneflow {

void AnchorTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_anchor_target_conf());

  EnrollInputBn("gt_boxes", false);
  EnrollInputBn("images", false);

  EnrollRepeatedOutputBn("rpn_labels", false);
  EnrollRepeatedOutputBn("rpn_bbox_targets", false);
  EnrollRepeatedOutputBn("rpn_bbox_weights", false);

  EnrollDataTmpBn("anchors");
  EnrollDataTmpBn("anchor_inds");
  EnrollDataTmpBn("anchor_boxes_inds");
  EnrollDataTmpBn("anchor_boxes_labels");
  EnrollDataTmpBn("max_overlaps");
  EnrollDataTmpBn("max_overlaps_with_gt_index");
  EnrollDataTmpBn("gt_boxes_inds");
}

const PbMessage& AnchorTargetOp::GetCustomizedConf() const {
  return this->op_conf().anchor_target_conf();
}

void AnchorTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const size_t num_layers = conf.anchor_generator_conf_size();
  CHECK_EQ(conf.rpn_labels_size(), num_layers);
  CHECK_EQ(conf.rpn_bbox_targets_size(), num_layers);
  CHECK_EQ(conf.rpn_bbox_weights_size(), num_layers);
  // input: gt_boxes (N, G, 4)
  const BlobDesc* gt_boxes_bd = GetBlobDesc4BnInOp("gt_boxes");
  CHECK(gt_boxes_bd->has_dim1_valid_num_field());
  CHECK(!gt_boxes_bd->has_instance_shape_field());
  // input: images (N, H, W, C)
  const BlobDesc* images_blob_desc = GetBlobDesc4BnInOp("images");
  const int64_t num_ims = gt_boxes_bd->shape().At(0);
  CHECK_EQ(images_blob_desc->shape().At(0), num_ims);
  CHECK_EQ(gt_boxes_bd->data_type(), images_blob_desc->data_type());
  const int64_t batch_h = images_blob_desc->shape().At(1);
  const int64_t batch_w = images_blob_desc->shape().At(2);
  int64_t num_anchors = 0;
  FOR_RANGE(size_t, layer, 0, num_layers) {
    const AnchorGeneratorConf& anchor_generator_conf = conf.anchor_generator_conf(layer);
    const int64_t num_anchors_per_cell =
        anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size();
    const float fm_stride = anchor_generator_conf.feature_map_stride();
    CHECK_GT(fm_stride, 0);
    const int64_t height = std::ceil(batch_h / fm_stride);
    const int64_t width = std::ceil(batch_w / fm_stride);
    CHECK_GT(height, 0);
    CHECK_GT(width, 0);
    num_anchors += height * width * num_anchors_per_cell;
    // repeat output: rpn_labels (N, H, W, A) int32_t
    BlobDesc* rpn_labels_bd = GetBlobDesc4BnInOp(GenRepeatedBn("rpn_labels", layer));
    rpn_labels_bd->set_data_type(DataType::kInt32);
    rpn_labels_bd->mut_shape() = Shape({num_ims, height, width, num_anchors_per_cell});
    rpn_labels_bd->set_has_instance_shape_field(images_blob_desc->has_instance_shape_field());
    // repeat output: rpn_bbox_targets (N, H, W, 4 * A) T
    BlobDesc* rpn_bbox_targets_bd = GetBlobDesc4BnInOp(GenRepeatedBn("rpn_bbox_targets", layer));
    rpn_bbox_targets_bd->set_data_type(gt_boxes_bd->data_type());
    rpn_bbox_targets_bd->mut_shape() = Shape({num_ims, height, width, num_anchors_per_cell * 4});
    rpn_bbox_targets_bd->set_has_instance_shape_field(images_blob_desc->has_instance_shape_field());
    // repeat output: rpn_bbox_inside_weights same as rpn_bbox_targets
    *GetBlobDesc4BnInOp(GenRepeatedBn("rpn_bbox_weights", layer)) = *rpn_bbox_targets_bd;
  }

  // data_tmp: anchors ((H1 * W1 + H2 * W2 + ...) * A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->set_data_type(gt_boxes_bd->data_type());
  anchors_blob_desc->mut_shape() = Shape({num_anchors, 4});
  anchors_blob_desc->mut_dim0_inner_shape() = Shape({1, num_anchors});
  anchors_blob_desc->set_has_dim0_valid_num_field(true);
  // data_tmp: anchors_inds (H1 * W1 + H2 * W2 + ...) * A) int32_t
  BlobDesc* anchors_inside_inds_bd = GetBlobDesc4BnInOp("anchors_inside_inds");
  anchors_inside_inds_bd->set_data_type(DataType::kInt32);
  anchors_inside_inds_bd->mut_shape() = Shape({num_anchors});
  anchors_inside_inds_bd->mut_dim0_inner_shape() = Shape({1, num_anchors});
  anchors_inside_inds_bd->set_has_dim0_valid_num_field(true);
  // data_tmp: anchor_boxes_inds same as anchors_inside_inds
  *GetBlobDesc4BnInOp("anchor_boxes_inds") = *anchors_inside_inds_bd;
  // data_tmp: anchor_boxes_labels (num_anchors) int32_t
  BlobDesc* anchor_boxes_labels_bd = GetBlobDesc4BnInOp("anchor_boxes_labels");
  anchor_boxes_labels_bd->set_data_type(DataType::kInt32);
  anchor_boxes_labels_bd->mut_shape() = Shape({num_anchors});
  // data_tmp: anchor_max_overlaps (num_anchors) float
  BlobDesc* max_overlaps_bd = GetBlobDesc4BnInOp("max_overlaps");
  max_overlaps_bd->set_data_type(DataType::kFloat);
  max_overlaps_bd->mut_shape() = Shape({num_anchors});
  // data_tmp: max_overlaps_with_gt_index (num_anchors) int32_t
  BlobDesc* max_overlaps_with_gt_index_bd = GetBlobDesc4BnInOp("max_overlaps_with_gt_index");
  max_overlaps_with_gt_index_bd->set_data_type(DataType::kInt32);
  max_overlaps_with_gt_index_bd->mut_shape() = Shape({num_anchors});
  // data_tmp: gt_boxes_inds (max_num_gt_boxes_per_im) int32_t
  BlobDesc* gt_boxes_inds_bd = GetBlobDesc4BnInOp("gt_boxes_inds");
  gt_boxes_inds_bd->set_data_type(DataType::kInt32);
  gt_boxes_inds_bd->mut_shape() = Shape({gt_boxes_bd->shape().At(1)});
}

void AnchorTargetOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("gt_boxes")->data_type());
}

REGISTER_CPU_OP(OperatorConf::kAnchorTargetConf, AnchorTargetOp);

}  // namespace oneflow
