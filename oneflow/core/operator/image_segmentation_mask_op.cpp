#include "oneflow/core/operator/image_segmentation_mask_op.h"

namespace oneflow {

void ImageSegmentationMaskOp::InitFromOpConf() {
  CHECK(device_type() == DeviceType::kCPU);
  if (!op_conf().image_segmentation_mask_conf().class_specific_mask()) { TODO(); }
  EnrollInputBn("roi_labels", false);
  EnrollInputBn("rois", false);
  EnrollInputBn("masks", false);
  EnrollDataTmpBn("padded_mask");
  EnrollDataTmpBn("im_mask");
  EnrollOutputBn("out", false);
}

const PbMessage& ImageSegmentationMaskOp::GetCustomizedConf() const {
  return op_conf().image_segmentation_mask_conf();
}

void ImageSegmentationMaskOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().image_segmentation_mask_conf();
  const BlobDesc* mask_blob_desc = GetBlobDesc4BnInOp("masks");
  const BlobDesc* roi_labels_blob_desc = GetBlobDesc4BnInOp("roi_labels");
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  CHECK_GE(conf.im_height(), mask_blob_desc->shape().At(2));
  CHECK_GE(conf.im_weight(), mask_blob_desc->shape().At(3));
  CHECK_EQ(mask_blob_desc->shape().At(0), roi_labels_blob_desc->shape().At(0));
  CHECK_EQ(mask_blob_desc->shape().At(0), rois_blob_desc->shape().At(0));
  CHECK_EQ(mask_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(roi_labels_blob_desc->shape().NumAxes(), 1);
  CHECK_EQ(rois_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(rois_blob_desc->shape().At(1), 5);

  // out: (R, im_height, im_weight)
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  Shape shape({mask_blob_desc->shape().At(0), conf.im_height(), conf.im_weight()});
  *out_blob_desc = *mask_blob_desc;
  out_blob_desc->mut_shape() = shape;
  out_blob_desc->set_data_type(DataType::kUInt8);

  // padded_mask: (M_w + 2, M + 2).
  // To work around an issue with cv2.resize, zero-pad the masks by 1 pixel
  BlobDesc* padded_mask_blob_desc = GetBlobDesc4BnInOp("padded_mask");
  padded_mask_blob_desc->mut_shape() =
      Shape({mask_blob_desc->shape().At(1) + 2, mask_blob_desc->shape().At(2) + 2});
  padded_mask_blob_desc->set_data_type(mask_blob_desc->data_type());

  // im_mask: (im_height, im_width)
  BlobDesc* im_mask_blob_desc = GetBlobDesc4BnInOp("im_mask");
  im_mask_blob_desc->mut_shape() = Shape({conf.im_height(), conf.im_weight()});
  im_mask_blob_desc->set_data_type(DataType::kUInt8);
}

void ImageSegmentationMaskOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("masks")->data_type());
}

REGISTER_OP(OperatorConf::kImageSegmentationMaskConf, ImageSegmentationMaskOp);

}  // namespace oneflow
