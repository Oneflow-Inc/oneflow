#include "oneflow/core/operator/iou_matrix_op.h"

namespace oneflow {

void IoUMatrixOp::InitFromOpConf() {
  CHECK(op_conf().has_iou_matrix_conf());
  EnrollInputBn("proposals", false);
  EnrollInputBn("gt_boxes", false);
  EnrollOutputBn("iou_matrix", false);
  EnrollOutputBn("iou_matrix_shape", false);
}

const PbMessage& IoUMatrixOp::GetCustomizedConf() const {
  return this->op_conf().iou_matrix_conf();
}

void IoUMatrixOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  // input: proposal (M, 4)
  const BlobDesc* proposals = GetBlobDesc4BnInOp("proposals");
  CHECK_EQ(proposals->shape().NumAxes(), 2);
  CHECK_EQ(proposals->shape().At(1), 4);
  const int32_t num_proposals = proposals->shape().At(0);
  // input: gt_boxes (N, G, 4)
  const BlobDesc* gt_boxes = GetBlobDesc4BnInOp("gt_boxes");
  CHECK_EQ(gt_boxes->shape().NumAxes(), 3);
  CHECK_EQ(gt_boxes->shape().At(2), 4);
  const int32_t num_imgs = gt_boxes->shape().At(0);
  const int32_t num_gt_boxes = gt_boxes->shape().At(1);
  // output: iou_matrix (N, M, G)
  BlobDesc* iou_matrix = GetBlobDesc4BnInOp("iou_matrix");
  iou_matrix->mut_shape() = Shape({num_imgs, num_proposals, num_gt_boxes});
  iou_matrix->set_data_type(DataType::kFloat);
  // output: iou_matrix_shape (N, 2)
  BlobDesc* iou_matrix_shape = GetBlobDesc4BnInOp("iou_matrix_shape");
  iou_matrix_shape->mut_shape() = Shape({num_imgs, 2});
  iou_matrix_shape->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kIouMatrixConf, IoUMatrixOp);

}  // namespace oneflow
