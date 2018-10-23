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
  // rois: (R, 5)
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  CHECK_EQ(rois_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(rois_blob_desc->shape().At(1), 5);
  CHECK(rois_blob_desc->has_dim0_inner_shape());
  // out: (R, C, pool_h, pool_w)
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *rois_blob_desc;
  out_blob_desc->mut_shape() =
      Shape({rois_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
             op_conf().roi_align_conf().pooled_h(), op_conf().roi_align_conf().pooled_w()});
  out_blob_desc->set_data_type(in_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kRoiAlignConf, RoIAlignOp);

}  // namespace oneflow
