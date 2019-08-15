#include "oneflow/core/operator/box_encode_op.h"

namespace oneflow {

void BoxEncodeOp::InitFromOpConf() {
  CHECK(op_conf().has_box_encode_conf());
  EnrollInputBn("ref_boxes", false);
  EnrollInputBn("boxes", false);
  EnrollOutputBn("boxes_delta", false);
}

const PbMessage& BoxEncodeOp::GetCustomizedConf() const {
  return this->op_conf().box_encode_conf();
}

void BoxEncodeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  // input: ref_boxes (N, 4)
  const BlobDesc* ref_boxes = GetBlobDesc4BnInOp("ref_boxes");
  CHECK_EQ(ref_boxes->shape().NumAxes(), 2);
  CHECK_EQ(ref_boxes->shape().At(1), 4);
  const bool dim0_varing = ref_boxes->has_dim0_valid_num_field();
  // input: boxes (N, 4)
  const BlobDesc* boxes = GetBlobDesc4BnInOp("boxes");
  CHECK_EQ(boxes->shape().NumAxes(), 2);
  CHECK_EQ(boxes->shape().At(1), 4);
  CHECK_EQ(ref_boxes->shape(), boxes->shape());
  CHECK_EQ(dim0_varing, boxes->has_dim0_valid_num_field());
  // output: boxes_delta (N, 4)
  BlobDesc* boxes_delta = GetBlobDesc4BnInOp("boxes_delta");
  boxes_delta->mut_shape() = ref_boxes->shape();
  boxes_delta->set_data_type(ref_boxes->data_type());
  if (dim0_varing) {
    boxes_delta->set_has_dim0_valid_num_field(true);
    boxes_delta->mut_dim0_inner_shape() = Shape({1, ref_boxes->shape().At(0)});
  }
}

REGISTER_OP(OperatorConf::kBoxEncodeConf, BoxEncodeOp);

}  // namespace oneflow
