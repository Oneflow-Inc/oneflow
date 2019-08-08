#include "oneflow/core/operator/identify_outside_anchors_op.h"

namespace oneflow {

void IdentifyOutsideAnchorsOp::InitFromOpConf() {
  CHECK(op_conf().has_identify_outside_anchors_conf());

  EnrollInputBn("anchors", false);
  EnrollInputBn("image_size", false);
  // Element which map to anchor outside image equal to 1, otherwise 0
  EnrollOutputBn("out", false);
}

void IdentifyOutsideAnchorsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input: anchors (R, 4)
  const BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  CHECK_EQ(anchors_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(anchors_blob_desc->shape().At(1), 4);
  // input: image_size (2)
  const BlobDesc* image_size_blob_desc = GetBlobDesc4BnInOp("image_size");
  CHECK_EQ(image_size_blob_desc->shape().NumAxes(), 1);
  CHECK_EQ(image_size_blob_desc->shape().At(0), 2);
  CHECK_EQ(image_size_blob_desc->data_type(), DataType::kInt32);

  // output: index (R, )
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->set_data_type(DataType::kInt8);
  out_blob_desc->mut_shape() = Shape({anchors_blob_desc->shape().At(0)});
  if (anchors_blob_desc->has_dim0_inner_shape()) {
    out_blob_desc->mut_dim0_inner_shape() = anchors_blob_desc->dim0_inner_shape();
    out_blob_desc->set_has_dim0_valid_num_field(anchors_blob_desc->has_dim0_inner_shape());
  }
}

}  // namespace oneflow
