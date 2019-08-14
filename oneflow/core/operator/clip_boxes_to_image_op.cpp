#include "oneflow/core/operator/clip_boxes_to_image_op.h"

namespace oneflow {

void ClipBoxesToImageOp::InitFromOpConf() {
  CHECK(op_conf().has_clip_boxes_to_image_conf());
  EnrollInputBn("boxes", false);
  EnrollInputBn("image_size", false);
  EnrollOutputBn("out", false);
}

const PbMessage& ClipBoxesToImageOp::GetCustomizedConf() const {
  return this->op_conf().clip_boxes_to_image_conf();
}

void ClipBoxesToImageOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input: boxes (R, 4)
  const BlobDesc* boxes = GetBlobDesc4BnInOp("boxes");
  CHECK_EQ(boxes->shape().NumAxes(), 2);
  CHECK_EQ(boxes->shape().At(1), 4);
  CHECK(!boxes->has_instance_shape_field());
  // input: image_size (2,)
  const BlobDesc* image_size = GetBlobDesc4BnInOp("image_size");
  CHECK_EQ(image_size->shape().NumAxes(), 1);
  CHECK_EQ(image_size->shape().At(0), 2);
  CHECK_EQ(image_size->data_type(), DataType::kInt32);
  // output: out (R, 4)
  *GetBlobDesc4BnInOp("out") = *boxes;
}

REGISTER_OP(OperatorConf::kClipBoxesToImageConf, ClipBoxesToImageOp);

}  // namespace oneflow
