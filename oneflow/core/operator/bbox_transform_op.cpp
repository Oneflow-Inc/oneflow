#include "oneflow/core/operator/bbox_transform_op.h"

namespace oneflow {

void BboxTransformOp::InitFromOpConf() {
  CHECK(op_conf().has_bbox_transform_conf());
  EnrollInputBn("bbox", false);
  EnrollInputBn("bbox_delta", false);
  EnrollOutputBn("out_bbox", false);
}

const PbMessage& BboxTransformOp::GetCustomizedConf() const {
  return op_conf().bbox_transform_conf();
}

void BboxTransformOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input: bbox (n, r, 4) or (r, 4)
  const BlobDesc* bbox_blob_desc = GetBlobDesc4BnInOp("bbox");
  // input: bbox_delta (n, r, 4) or (n, r, 4*c) or (r, 4) or (r, 4*c)
  const BlobDesc* bbox_delta_blob_desc = GetBlobDesc4BnInOp("bbox_delta");
  CHECK(!bbox_blob_desc->has_instance_shape_field());
  CHECK(!bbox_delta_blob_desc->has_instance_shape_field());
  int64_t num_axes = bbox_blob_desc->shape().NumAxes();
  CHECK_EQ(bbox_delta_blob_desc->shape().NumAxes(), num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes - 1) {
    CHECK_EQ(bbox_blob_desc->shape().At(i), bbox_delta_blob_desc->shape().At(i));
  }
  CHECK_EQ(
      bbox_blob_desc->shape().At(num_axes - 1) % bbox_delta_blob_desc->shape().At(num_axes - 1), 0);

  // output: out_bbox (n, r, 4) or (n, r, 4*c) or (r, 4) or (r, 4*c)
  *GetBlobDesc4BnInOp("out_bbox") = *bbox_delta_blob_desc;
  if (bbox_blob_desc->has_dim0_inner_shape()) {
    CHECK(bbox_delta_blob_desc->has_dim0_inner_shape());
    CHECK_EQ(bbox_blob_desc->dim0_inner_shape(), bbox_delta_blob_desc->dim0_inner_shape());
    CHECK_EQ(bbox_blob_desc->has_dim0_valid_num_field(),
             bbox_delta_blob_desc->has_dim0_valid_num_field());
  }
  CHECK_EQ(bbox_blob_desc->has_dim1_valid_num_field(),
           bbox_delta_blob_desc->has_dim0_valid_num_field());
}

REGISTER_OP(OperatorConf::kBboxTransformConf, BboxTransformOp);

}  // namespace oneflow
