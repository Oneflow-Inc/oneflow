#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

void PoolingOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
  VirtualEnrollDataTmpBn();
}

void PoolingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  int64_t shape_h =
      (in_blob_desc->shape().At(2) + 2 * GetInt32FromSpecialConf("pad_h")
       - GetInt32FromSpecialConf("kernel_h"))
          / GetInt32FromSpecialConf("stride_h")
      + 1;
  int64_t shape_w =
      (in_blob_desc->shape().At(3) + 2 * GetInt32FromSpecialConf("pad_w")
       - GetInt32FromSpecialConf("kernel_w"))
          / GetInt32FromSpecialConf("stride_w")
      + 1;
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1), shape_h,
             shape_w});
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id_field(in_blob_desc->has_data_id_field());

  VirtualInferDataTmpBlobDesc(GetBlobDesc4BnInOp);
}

}  // namespace oneflow
