#include "oneflow/core/operator/roi_align_op.h"

namespace oneflow {

void RoIAlignOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollInputBn("rois", false);
  EnrollOutputBn("out");
}

const PbMessage& RoIAlignOp::GetCustomizedConf() const { return op_conf().roi_align_conf(); }

void RoIAlignOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  if (op_conf().roi_align_conf().data_format() != "channels_first") { UNIMPLEMENTED(); }
  // in: feature map (N, C, H, W)
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  // rois: (N, R, 4)
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  CHECK_EQ(rois_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(rois_blob_desc->shape().At(0), in_blob_desc->shape().At(0));
  CHECK_EQ(rois_blob_desc->shape().At(2), 4);
  // out: (N * R, C, pool_h, pool_w)
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(
      {in_blob_desc->shape().At(0) * rois_blob_desc->shape().At(1), in_blob_desc->shape().At(1),
       op_conf().roi_align_conf().pooled_h(), op_conf().roi_align_conf().pooled_w()});
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());
  out_blob_desc->set_has_col_num_field(in_blob_desc->has_col_num_field());
}

REGISTER_OP(OperatorConf::kRoiAlignConf, RoIAlignOp);

}  // namespace oneflow
