#include "oneflow/core/operator/anchor_target_op.h"

namespace oneflow {

void AnchorTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_anchor_target_conf());

  EnrollInputBn("images", false);
  EnrollInputBn("gt_boxes", false);

  EnrollRepeatedOutputBn("regression_targets", false);
  EnrollRepeatedOutputBn("regression_weights", false);
  EnrollRepeatedOutputBn("class_labels", false);
  EnrollRepeatedOutputBn("class_weights", false);

  EnrollDataTmpBn("anchors");
  EnrollDataTmpBn("anchor_inds");
  EnrollDataTmpBn("anchor_labels");
  EnrollDataTmpBn("anchor_max_overlaps");
  EnrollDataTmpBn("anchor_best_match_gt");
}

const PbMessage& AnchorTargetOp::GetCustomizedConf() const {
  return this->op_conf().anchor_target_conf();
}

void AnchorTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const size_t num_layers = conf.anchor_generator_conf_size();
  CHECK_EQ(conf.regression_targets_size(), num_layers);
  CHECK_EQ(conf.regression_weights_size(), num_layers);
  CHECK_EQ(conf.class_labels_size(), num_layers);
  CHECK_EQ(conf.class_weights_size(), num_layers);
  // input: gt_boxes (N, G, 4)
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  CHECK(!gt_boxes_blob_desc->has_dim0_valid_num_field());
  CHECK(gt_boxes_blob_desc->has_dim1_valid_num_field());
  CHECK(!gt_boxes_blob_desc->has_instance_shape_field());
  // input: images (N, H, W, C)
  const BlobDesc* images_blob_desc = GetBlobDesc4BnInOp("images");
  const int64_t num_images = images_blob_desc->shape().At(0);
  DataType data_type = images_blob_desc->data_type();
  CHECK(!images_blob_desc->has_dim0_valid_num_field());
  CHECK_EQ(gt_boxes_blob_desc->shape().At(0), num_images);
  CHECK_EQ(gt_boxes_blob_desc->data_type(), data_type);
  const int64_t batch_height = images_blob_desc->shape().At(1);
  const int64_t batch_width = images_blob_desc->shape().At(2);

  int64_t total_num_anchors = 0;
  FOR_RANGE(size_t, layer, 0, num_layers) {
    const AnchorGeneratorConf& anchor_generator_conf = conf.anchor_generator_conf(layer);
    const int64_t num_anchors_per_cell =
        anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size();
    const float fm_stride = anchor_generator_conf.feature_map_stride();
    CHECK_GT(fm_stride, 0);
    const int64_t height = std::ceil(batch_height / fm_stride);
    const int64_t width = std::ceil(batch_width / fm_stride);
    CHECK_GT(height, 0);
    CHECK_GT(width, 0);
    total_num_anchors += height * width * num_anchors_per_cell;
    // repeat output: class_labels (N, h_i, w_i, A) int32_t
    BlobDesc* class_labels_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn("class_labels", layer));
    class_labels_blob_desc->set_data_type(DataType::kInt32);
    class_labels_blob_desc->mut_shape() = Shape({num_images, height, width, num_anchors_per_cell});
    class_labels_blob_desc->set_has_instance_shape_field(
        images_blob_desc->has_instance_shape_field());
    // repeat output: class_weights (N, h_i, w_i, A) T
    BlobDesc* class_weights_blob_desc = GetBlobDesc4BnInOp(GenRepeatedBn("class_weights", layer));
    *class_weights_blob_desc = *class_labels_blob_desc;
    class_weights_blob_desc->set_data_type(data_type);
    // repeat output: regression_targets (N, H, W, 4 * A) T
    BlobDesc* regression_targets_blob_desc =
        GetBlobDesc4BnInOp(GenRepeatedBn("regression_targets", layer));
    *regression_targets_blob_desc = *class_weights_blob_desc;
    regression_targets_blob_desc->mut_shape() =
        Shape({num_images, height, width, num_anchors_per_cell * 4});
    // repeat output: regression_weights same as regression_targets
    *GetBlobDesc4BnInOp(GenRepeatedBn("regression_weights", layer)) = *regression_targets_blob_desc;
  }

  // data_tmp: anchors ((H1 * W1 + H2 * W2 + ...) * A, 4) T
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->set_data_type(gt_boxes_blob_desc->data_type());
  anchors_blob_desc->mut_shape() = Shape({total_num_anchors, 4});
  anchors_blob_desc->mut_dim0_inner_shape() = Shape({1, total_num_anchors});
  anchors_blob_desc->set_has_dim0_valid_num_field(true);
  // data_tmp: anchors_inds (H1 * W1 + H2 * W2 + ...) * A) int32_t
  BlobDesc* anchor_inds_blob_desc = GetBlobDesc4BnInOp("anchor_inds");
  *anchor_inds_blob_desc = *anchors_blob_desc;
  anchor_inds_blob_desc->mut_shape() = Shape({total_num_anchors});
  anchor_inds_blob_desc->set_data_type(DataType::kInt32);
  // data_tmp: anchor_labels has the same shape as anchors_inds
  *GetBlobDesc4BnInOp("anchor_labels") = *anchor_inds_blob_desc;
  // data_tmp: anchor_max_overlaps has the same shape as anchors_inds
  BlobDesc* anchor_max_overlaps_blob_desc = GetBlobDesc4BnInOp("anchor_max_overlaps");
  *anchor_max_overlaps_blob_desc = *anchor_inds_blob_desc;
  anchor_max_overlaps_blob_desc->set_data_type(DataType::kFloat);
  // data_tmp: anchor_best_match_gt has the same shape as anchors_inds
  BlobDesc* anchor_best_match_gt_blob_desc = GetBlobDesc4BnInOp("anchor_best_match_gt");
  *anchor_best_match_gt_blob_desc = *anchor_inds_blob_desc;
  anchor_best_match_gt_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_CPU_OP(OperatorConf::kAnchorTargetConf, AnchorTargetOp);

}  // namespace oneflow
