#include "oneflow/core/operator/mask_target_op.h"

namespace oneflow {

void MaskTargetOp::InitFromOpConf() {
  CHECK_EQ(this->device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_mask_target_conf());
  // Enroll input
  EnrollInputBn("in_rois", false);
  EnrollInputBn("in_labels", false);
  EnrollInputBn("gt_segm_polygon_lists", false);
  // Enroll output
  EnrollOutputBn("mask_rois", false);
  EnrollOutputBn("masks", false);
  // Enroll data tmp
  EnrollDataTmpBn("gt_segm_bboxes");
  // Enroll const buf
  EnrollConstBufBn("mask_ignore_labels");
}

const PbMessage& MaskTargetOp::GetCustomizedConf() const { return op_conf().mask_target_conf(); }

void MaskTargetOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const MaskTargetOpConf& conf = op_conf().mask_target_conf();
  CHECK_GT(conf.mask_h(), 0);
  CHECK_GT(conf.mask_w(), 0);
  CHECK_GT(conf.num_classes(), 0);
  // input: in_rois (R, 5) T
  const BlobDesc* in_rois = GetBlobDesc4BnInOp("in_rois");
  // input: in_labels (R) int32_t
  const BlobDesc* in_labels = GetBlobDesc4BnInOp("in_labels");
  // input: gt_segm_polygon_lists (N,G,B) byte
  const BlobDesc* gt_segm_polygon_lists = GetBlobDesc4BnInOp("gt_segm_polygon_lists");
  CHECK_EQ(in_rois->shape().NumAxes(), 2);
  CHECK_EQ(in_labels->shape().NumAxes(), 1);
  CHECK_EQ(gt_segm_polygon_lists->shape().NumAxes(), 3);
  int64_t R = in_rois->shape().At(0);
  int64_t N = gt_segm_polygon_lists->shape().At(0);
  int64_t G = gt_segm_polygon_lists->shape().At(1);
  int64_t class_num = conf.num_classes();
  DataType data_type = in_rois->data_type();

  CHECK_EQ(in_rois->has_dim0_valid_num_field(), in_labels->has_dim0_valid_num_field());
  auto CheckDim0InnerShapeIfNeed = [](const BlobDesc* blob_desc) {
    if (blob_desc->has_dim0_valid_num_field()) {
      CHECK(blob_desc->has_dim0_inner_shape());
      CHECK_EQ(blob_desc->dim0_inner_shape().At(0), 1);
      CHECK_EQ(blob_desc->dim0_inner_shape().NumAxes(), 2);
    }
  };
  CheckDim0InnerShapeIfNeed(in_rois);
  CheckDim0InnerShapeIfNeed(in_labels);
  CHECK_EQ(in_rois->shape().At(1), 5);
  CHECK_EQ(in_labels->shape().At(0), R);
  CHECK_EQ(in_labels->data_type(), DataType::kInt32);
  CHECK_EQ(GetSizeOfDataType(gt_segm_polygon_lists->data_type()), 1);

  // output: mask_rois (R, 5) T
  BlobDesc* mask_rois = GetBlobDesc4BnInOp("mask_rois");
  mask_rois->mut_shape() = Shape({R, 5});
  mask_rois->set_data_type(data_type);
  mask_rois->set_has_dim0_valid_num_field(true);
  mask_rois->mut_dim0_inner_shape() = Shape({1, R});
  // output: masks (R, class_num , mask_h, mask_w) T
  BlobDesc* masks = GetBlobDesc4BnInOp("masks");
  masks->mut_shape() = Shape({R, class_num, conf.mask_h(), conf.mask_w()});
  masks->set_data_type(data_type);
  masks->set_has_dim0_valid_num_field(true);
  masks->mut_dim0_inner_shape() = Shape({1, R});
  // data tmp: mask_boxes (N,G,4) float
  BlobDesc* gt_segm_bboxes = GetBlobDesc4BnInOp("gt_segm_bboxes");
  gt_segm_bboxes->mut_shape() = Shape({N, G, 4});
  gt_segm_bboxes->set_data_type(DataType::kFloat);
  gt_segm_bboxes->set_has_dim1_valid_num_field(true);
  // const buf: mask_ignore_labels(class_num, mask_h, mask_w)
  BlobDesc* mask_ignore_labels = GetBlobDesc4BnInOp("mask_ignore_labels");
  mask_ignore_labels->mut_shape() = Shape({class_num, conf.mask_h(), conf.mask_w()});
  mask_ignore_labels->set_data_type(data_type);
}

REGISTER_OP(OperatorConf::kMaskTargetConf, MaskTargetOp);

}  // namespace oneflow
