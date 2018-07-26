#include "oneflow/core/operator/roi_pooling_op.h"

namespace oneflow {

void RoIPoolingOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollInputBn("rois", false);
  EnrollOutputBn("out");
  EnrollDataTmpBn("argmax");
}

const PbMessage& RoIPoolingOp::GetCustomizedConf() const { return op_conf().roi_pooling_conf(); }

void RoIPoolingOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  // rois
  const BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  CHECK_EQ(rois_blob_desc->shape().NumAxes(), 3);
  CHECK_EQ(rois_blob_desc->shape().At(0), in_blob_desc->shape().At(0));
  CHECK_EQ(rois_blob_desc->shape().At(2), 4);
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(
      {in_blob_desc->shape().At(0) * rois_blob_desc->shape().At(1), in_blob_desc->shape().At(1),
       op_conf().roi_pooling_conf().pooled_h(), op_conf().roi_pooling_conf().pooled_w()});
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());
  out_blob_desc->set_has_col_num_field(in_blob_desc->has_col_num_field());
  CHECK_GE(op_conf().roi_pooling_conf().pooled_h(), 0);
  CHECK_GE(op_conf().roi_pooling_conf().pooled_w(), 0);
  // argmax
  BlobDesc* argmax_blob_desc = GetBlobDesc4BnInOp("argmax");
  argmax_blob_desc->mut_shape() = out_blob_desc->shape();
  argmax_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kRoiPoolingConf, RoIPoolingOp);

}  // namespace oneflow
