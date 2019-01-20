#include "oneflow/core/operator/top_k_op.h"

namespace oneflow {

void TopKOp::InitFromOpConf() {
  CHECK(op_conf().has_top_k_conf());
  EnrollInputBn("in", false);
  EnrollFwBufBn("fw_buf");
  EnrollOutputBn("out", false);
}

void TopKOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  const TopKOpConf& conf = op_conf().top_k_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_GE(conf.k(), 1);
  CHECK_LE(conf.k(), in_blob_desc->shape().dim_vec().back());
  // fw_buf
  BlobDesc* fw_buf_blob_desc = GetBlobDesc4BnInOp("fw_buf");
  fw_buf_blob_desc->mut_shape() = Shape({in_blob_desc->shape().dim_vec().back()});
  fw_buf_blob_desc->set_data_type(DataType::kInt32);
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape().Set(in_blob_desc->shape().NumAxes() - 1, conf.k());
  out_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kTopKConf, TopKOp);

}  // namespace oneflow
