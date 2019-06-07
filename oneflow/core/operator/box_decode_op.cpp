#include "oneflow/core/operator/box_decode_op.h"

namespace oneflow {

void BoxDecodeOp::InitFromOpConf() {
  CHECK(op_conf().has_box_decode_conf());
  EnrollInputBn("ref_boxes", false);
  EnrollInputBn("boxes_delta", false);
  EnrollOutputBn("boxes", false);
}

const PbMessage& BoxDecodeOp::GetCustomizedConf() const {
  return this->op_conf().box_decode_conf();
}

void BoxDecodeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  // input: ref_boxes (N, 4)
  const BlobDesc* ref_boxes = GetBlobDesc4BnInOp("ref_boxes");
  CHECK_EQ(ref_boxes->shape().NumAxes(), 2);
  CHECK_EQ(ref_boxes->shape().At(1), 4);
  const bool dim0_varing = ref_boxes->has_dim0_valid_num_field();
  // input: boxes_delta (N, 4)
  const BlobDesc* boxes_delta = GetBlobDesc4BnInOp("boxes_delta");
  CHECK_EQ(boxes_delta->shape().NumAxes(), 2);
  CHECK_EQ(boxes_delta->shape().At(1), 4);
  CHECK_EQ(ref_boxes->shape(), boxes_delta->shape());
  CHECK_EQ(dim0_varing, boxes_delta->has_dim0_valid_num_field());
  // output: boxes (N, 4)
  BlobDesc* boxes = GetBlobDesc4BnInOp("boxes_delta");
  boxes->mut_shape() = ref_boxes->shape();
  boxes->set_data_type(ref_boxes->data_type());
  if (dim0_varing) {
    boxes->set_has_dim0_valid_num_field(true);
    boxes->mut_dim0_inner_shape() = Shape({1, ref_boxes->shape().At(0)});
  }
}

REGISTER_OP(OperatorConf::kBoxDecodeConf, BoxDecodeOp);

}  // namespace oneflow
