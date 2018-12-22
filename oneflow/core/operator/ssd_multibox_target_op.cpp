#include "oneflow/core/operator/ssd_multibox_target_op.h"

namespace oneflow {

void SSDMultiboxTargetOp::InitFromOpConf() {
  // Enroll input
  EnrollInputBn("bbox", false);
  EnrollInputBn("gt_boxes", false);
  EnrollInputBn("gt_labels", false);
  // Enroll output
  EnrollOutputBn("sampled_indices", false);
  EnrollOutputBn("positive_sampled_indices", false);
  EnrollOutputBn("bbox_labels", false);
  EnrollOutputBn("bbox_deltas", false);
  EnrollOutputBn("bbox_inside_weights", false);
  EnrollOutputBn("bbox_outside_weights", false);
}

const PbMessage& SSDMultiboxTargetOp::GetCustomizedConf() const {
  return op_conf().ssd_multibox_target_conf();
}

void SSDMultiboxTargetOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const SSDMultiBoxTargetOpConf& conf = op_conf().ssd_multibox_target_conf();
  // TODO: Check conf
  // input: bbox (n, r, 4) T
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  // input: gt_boxes (n, g, 4) T
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  // input: gt_labels (n, g) int32_t
  const BlobDesc* gt_labels_blob_desc = GetBlobDesc4BnInOp("gt_labels");
  int64_t num_images = bbox_blob_desc->shape().At(0);
  CHECK_EQ(num_images, gt_boxes_blob_desc->shape().At(0));
  CHECK_EQ(num_images, gt_labels_blob_desc->shape().At(0));
  int64_t num_boxes = bbox_blob_desc->shape().At(1);
  int64_t max_num_gt_boxes = gt_boxes_blob_desc->shape().At(1);
  CHECK_EQ(max_num_gt_boxes, gt_labels_blob_desc->shape().At(1));
  CHECK_EQ(bbox_blob_desc->data_type(), gt_boxes_blob_desc->data_type());

  // output: sampled_indices (n, r) int32_t dim1 varying
  BlobDesc* sampled_indices_blob_desc = GetBlobDesc4BnInOp("sampled_indices");
  sampled_indices_blob_desc->mut_shape() = Shape({num_images, num_boxes});
  sampled_indices_blob_desc->set_data_type(DataType::kInt32);
  sampled_indices_blob_desc->set_has_dim1_valid_num_field(true);
  // output: positive_sampled_indices (n, r) int32_t dim1 varying
  *GetBlobDesc4BnInOp("positive_sampled_indices") = *sampled_indices_blob_desc;
  // output: bbox_labels (n, r) int32_t
  *GetBlobDesc4BnInOp("bbox_labels") = *sampled_indices_blob_desc;
  // output: bbox_deltas (n, r, 4) T
  BlobDesc* bbox_deltas_blob_desc = GetBlobDesc4BnInOp("bbox_deltas");
  bbox_deltas_blob_desc->mut_shape() = Shape({num_images, num_boxes, 4});
  bbox_deltas_blob_desc->set_data_type(bbox_blob_desc->data_type());
  bbox_deltas_blob_desc->set_has_dim1_valid_num_field(true);
  // output: bbox_inside_weights (n, r, 4) T
  *GetBlobDesc4BnInOp("bbox_inside_weights") = *bbox_deltas_blob_desc;
  // output: bbox_outside_weights (n, r, 4) T
  *GetBlobDesc4BnInOp("bbox_outside_weights") = *bbox_deltas_blob_desc;
}

REGISTER_CPU_OP(OperatorConf::kSsdMultiboxTargetConf, SSDMultiboxTargetOp);

}  // namespace oneflow