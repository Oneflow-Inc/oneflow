#include "oneflow/core/operator/image_segmentation_mask_op.h"

namespace oneflow {

void ImageSegmentationMaskOp::InitFromOpConf() {
  CHECK(device_type() == DeviceType::kCPU);
  if (!op_conf().image_segmentation_mask_conf().class_specific_mask()) { TODO(); }
  EnrollInputBn("bbox", false);
  EnrollInputBn("mask", false);
  EnrollInputBn("image_size", false);
  EnrollOutputBn("out", false);
  EnrollDataTmpBn("padded_mask");
}

const PbMessage& ImageSegmentationMaskOp::GetCustomizedConf() const {
  return op_conf().image_segmentation_mask_conf();
}

void ImageSegmentationMaskOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().image_segmentation_mask_conf();
  // mask (R, m_h, m_w) T
  const BlobDesc* mask_blob_desc = GetBlobDesc4BnInOp("mask");
  // bbox (R, 4) T
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  // image_size (N, 2)
  const BlobDesc* image_size_blob_desc = GetBlobDesc4BnInOp("image_size");

  const int64_t num_masks = mask_blob_desc->shape().At(0);
  const int64_t num_bboxes = bbox_blob_desc->shape().At(0);
  const int64_t mask_h = mask_blob_desc->shape().At(1);
  const int64_t mask_w = mask_blob_desc->shape().At(2);
  DataType data_type = mask_blob_desc->data_type();
  CHECK_EQ(num_masks, num_bboxes);
  CHECK_EQ(mask_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(bbox_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(bbox_blob_desc->shape().At(1), 4);
  CHECK_EQ(image_size_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(image_size_blob_desc->shape().At(1), 2);
  CHECK_EQ(mask_blob_desc->has_dim0_valid_num_field(), bbox_blob_desc->has_dim0_valid_num_field());
  CHECK(mask_blob_desc->has_record_id_in_device_piece_field());
  CHECK(bbox_blob_desc->has_record_id_in_device_piece_field());
  CHECK_EQ(mask_blob_desc->data_type(), bbox_blob_desc->data_type());
  CHECK_EQ(image_size_blob_desc->data_type(), DataType::kInt32);

  // out: (R, im_max_h, im_max_w) UInt8
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({num_masks, conf.image_max_height(), conf.image_max_width()});
  out_blob_desc->set_data_type(DataType::kUInt8);
  out_blob_desc->mut_dim0_inner_shape() = mask_blob_desc->dim0_inner_shape();
  out_blob_desc->set_has_dim0_valid_num_field(mask_blob_desc->has_dim0_valid_num_field());

  // padded_mask: (R, m_h + padding * 2, m_w + padding * 2)
  // To work around an issue with cv2.resize, zero-pad the masks by 1 pixel
  BlobDesc* padded_mask_blob_desc = GetBlobDesc4BnInOp("padded_mask");
  padded_mask_blob_desc->mut_shape() =
      Shape({num_masks, mask_h + conf.padding() * 2, mask_w + conf.padding() * 2});
  padded_mask_blob_desc->set_data_type(mask_blob_desc->data_type());
}

void ImageSegmentationMaskOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("mask")->data_type());
}

REGISTER_CPU_OP(OperatorConf::kImageSegmentationMaskConf, ImageSegmentationMaskOp);

}  // namespace oneflow
