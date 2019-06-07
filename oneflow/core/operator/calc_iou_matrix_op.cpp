#include "oneflow/core/operator/calc_iou_matrix_op.h"

namespace oneflow {

void CalcIoUMatrixOp::InitFromOpConf() {
  CHECK(op_conf().has_calc_iou_matrix_conf());
  EnrollInputBn("boxes1", false);
  EnrollInputBn("boxes2", false);
  EnrollOutputBn("iou_matrix", false);
}

const PbMessage& CalcIoUMatrixOp::GetCustomizedConf() const {
  return this->op_conf().calc_iou_matrix_conf();
}

void CalcIoUMatrixOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input: boxes1 (M, 4)
  const BlobDesc* boxes1 = GetBlobDesc4BnInOp("boxes1");
  CHECK_EQ(boxes1->shape().NumAxes(), 2);
  CHECK_EQ(boxes1->shape().At(1), 4);
  const int32_t num_boxes1 = boxes1->shape().At(0);
  // input: boxes2 (G, 4)
  const BlobDesc* boxes2 = GetBlobDesc4BnInOp("boxes2");
  CHECK_EQ(boxes2->shape().NumAxes(), 2);
  CHECK_EQ(boxes2->shape().At(1), 4);
  const int32_t num_boxes2 = boxes2->shape().At(0);
  // output: iou_matrix (M, G)
  BlobDesc* iou_matrix = GetBlobDesc4BnInOp("iou_matrix");
  iou_matrix->mut_shape() = Shape({num_boxes1, num_boxes2});
  iou_matrix->set_data_type(DataType::kFloat);
  iou_matrix->set_has_dim0_valid_num_field(boxes1->has_dim0_valid_num_field());
  iou_matrix->mut_dim0_inner_shape() = Shape({1, num_boxes1});
  iou_matrix->set_has_instance_shape_field(boxes2->has_dim0_valid_num_field());
}

REGISTER_OP(OperatorConf::kCalcIouMatrixConf, CalcIoUMatrixOp);

}  // namespace oneflow
