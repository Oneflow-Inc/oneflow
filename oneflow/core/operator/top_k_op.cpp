#include "oneflow/core/operator/top_k_op.h"

namespace oneflow {

void TopKOp::InitFromOpConf() {
  CHECK(op_conf().has_top_k_conf());
  const TopKOpConf& conf = op_conf().top_k_conf();
  CHECK(conf.has_values() || conf.has_indices());
  EnrollInputBn("in", false);
  EnrollFwBufBn("fw_buf");
  if (conf.has_values()) { EnrollOutputBn("values", false); }
  if (conf.has_indices()) { EnrollOutputBn("indices", false); }
}

void TopKOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  const TopKOpConf& conf = op_conf().top_k_conf();
  CHECK_GE(conf.k(), 1);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  std::vector<int64_t> out_shape_dim_vec = in->shape().dim_vec();
  CHECK_LE(conf.k(), out_shape_dim_vec.back());
  out_shape_dim_vec.back() = conf.k();
  BlobDesc* fw_buf = GetBlobDesc4BnInOp("fw_buf");
  fw_buf->mut_shape() = Shape({in->shape().dim_vec().back()});
  fw_buf->set_data_type(DataType::kInt64);
  if (conf.has_values()) {
    BlobDesc* values = GetBlobDesc4BnInOp("values");
    *values = *in;
    values->mut_shape() = Shape(out_shape_dim_vec);
  }
  if (conf.has_indices()) {
    BlobDesc* indices = GetBlobDesc4BnInOp("indices");
    *indices = *in;
    indices->mut_shape() = Shape(out_shape_dim_vec);
    indices->set_data_type(DataType::kInt64);
  }
}

REGISTER_OP(OperatorConf::kTopKConf, TopKOp);

}  // namespace oneflow
