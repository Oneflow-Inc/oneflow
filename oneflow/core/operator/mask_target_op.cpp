#include "oneflow/core/operator/mask_target_op.h"

namespace oneflow {

void MaskTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_mask_target_conf());
  // Enroll input
  EnrollInputBn("rois", false);
  EnrollInputBn("labels", false);
  EnrollInputBn("gt_segm_polygon_lists", false);
  EnrollInputBn("im_scale", false);
  // Enroll output
  EnrollOutputBn("mask_rois", false);
  EnrollOutputBn("masks", false);
  EnrollOutputBn("mask_labels", false);
  // Enroll data tmp
  EnrollDataTmpBn("gt_segm_bboxes");
}

const PbMessage& MaskTargetOp::GetCustomizedConf() const { return op_conf().mask_target_conf(); }

void MaskTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const MaskTargetOpConf& conf = op_conf().mask_target_conf();
  CHECK_GT(conf.mask_height(), 0);
  CHECK_GT(conf.mask_width(), 0);
  CHECK_GT(conf.num_classes(), 0);
  // input: rois (R, 5) T
  const BlobDesc* rois = GetBlobDesc4BnInOp("rois");
  // input: labels (R) int32_t
  const BlobDesc* labels = GetBlobDesc4BnInOp("labels");
  // input: gt_segm_polygon_lists (N,G,B) byte
  const BlobDesc* gt_segm_polygon_lists = GetBlobDesc4BnInOp("gt_segm_polygon_lists");
  // input: im_scale (N) T
  const BlobDesc* im_scale = GetBlobDesc4BnInOp("im_scale");
  CHECK_EQ(rois->shape().NumAxes(), 2);
  CHECK_EQ(labels->shape().NumAxes(), 1);
  CHECK_EQ(gt_segm_polygon_lists->shape().NumAxes(), 3);
  CHECK_EQ(im_scale->shape().NumAxes(), 1);
  int64_t R = rois->shape().At(0);
  int64_t N = gt_segm_polygon_lists->shape().At(0);
  int64_t G = gt_segm_polygon_lists->shape().At(1);
  DataType data_type = rois->data_type();

  const bool input_has_record_id = rois->has_record_id_in_device_piece_field();
  CHECK_EQ(rois->has_dim0_valid_num_field(), labels->has_dim0_valid_num_field());
  auto CheckDim0InnerShapeIfNeed = [](const BlobDesc* blob_desc) {
    if (blob_desc->has_dim0_valid_num_field()) {
      CHECK(blob_desc->has_dim0_inner_shape());
      CHECK_EQ(blob_desc->dim0_inner_shape().At(0), 1);
      CHECK_EQ(blob_desc->dim0_inner_shape().NumAxes(), 2);
    }
  };
  CheckDim0InnerShapeIfNeed(rois);
  CheckDim0InnerShapeIfNeed(labels);
  CHECK_EQ(labels->has_record_id_in_device_piece_field(), input_has_record_id);
  CHECK_EQ(rois->shape().At(1), 5);
  CHECK_EQ(labels->shape().At(0), R);
  CHECK_EQ(labels->data_type(), DataType::kInt32);
  CHECK_EQ(GetSizeOfDataType(gt_segm_polygon_lists->data_type()), 1);
  CHECK_EQ(im_scale->data_type(), rois->data_type());
  CHECK_EQ(im_scale->shape().At(0), gt_segm_polygon_lists->shape().At(0));

  // output: mask_rois (R, 5) T
  BlobDesc* mask_rois = GetBlobDesc4BnInOp("mask_rois");
  mask_rois->mut_shape() = Shape({R, 5});
  mask_rois->set_data_type(data_type);
  mask_rois->set_has_dim0_valid_num_field(true);
  mask_rois->mut_dim0_inner_shape() = Shape({1, R});
  mask_rois->set_has_record_id_in_device_piece_field(input_has_record_id);
  // output: masks (R, mask_h, mask_w) T
  BlobDesc* masks = GetBlobDesc4BnInOp("masks");
  masks->mut_shape() = Shape({R, conf.mask_height(), conf.mask_width()});
  masks->set_data_type(DataType::kInt32);
  masks->set_has_dim0_valid_num_field(true);
  masks->mut_dim0_inner_shape() = Shape({1, R});
  masks->set_has_record_id_in_device_piece_field(input_has_record_id);
  // output: mask_labels (R, mask_h, mask_w) T
  BlobDesc* mask_labels = GetBlobDesc4BnInOp("mask_labels");
  mask_labels->mut_shape() = Shape({R});
  mask_labels->set_data_type(DataType::kInt32);
  mask_labels->set_has_dim0_valid_num_field(true);
  mask_labels->mut_dim0_inner_shape() = Shape({1, R});
  mask_labels->set_has_record_id_in_device_piece_field(input_has_record_id);
  // data tmp: mask_boxes (N,G,4) float
  BlobDesc* gt_segm_bboxes = GetBlobDesc4BnInOp("gt_segm_bboxes");
  gt_segm_bboxes->mut_shape() = Shape({N, G, 4});
  gt_segm_bboxes->set_data_type(DataType::kFloat);
  gt_segm_bboxes->set_has_dim1_valid_num_field(true);
}

void MaskTargetOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("rois")->data_type());
}

REGISTER_CPU_OP(OperatorConf::kMaskTargetConf, MaskTargetOp);

}  // namespace oneflow
