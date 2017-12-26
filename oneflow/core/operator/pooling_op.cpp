#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

void PoolingOp::InitFromOpConf() {
  CHECK(op_conf().has_pooling_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (op_conf().pooling_conf().pool() == PoolingOpConf::kMax)
    EnrollDataTmpBn("idx");
}

const PbMessage& PoolingOp::GetSpecialConf() const {
  return op_conf().pooling_conf();
}

void PoolingOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const PoolingOpConf& conf = op_conf().pooling_conf();
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  int64_t shape_h =
      (in_blob_desc->shape().At(2) + 2 * conf.pad_h() - conf.kernel_h())
          / conf.stride_h()
      + 1;
  int64_t shape_w =
      (in_blob_desc->shape().At(3) + 2 * conf.pad_w() - conf.kernel_w())
          / conf.stride_w()
      + 1;
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1), shape_h,
             shape_w});
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());

  // idx
  if (conf.pool() == PoolingOpConf::kMax) {
    BlobDesc* idx_blob_desc = GetBlobDesc4BnInOp("idx");
    idx_blob_desc->mut_shape() = out_blob_desc->shape();
    idx_blob_desc->set_data_type(DataType::kUInt32);
    idx_blob_desc->set_has_data_id(false);
  }
}

REGISTER_OP(OperatorConf::kPoolingConf, PoolingOp);

}  // namespace oneflow
