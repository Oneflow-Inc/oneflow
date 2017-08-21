#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

void PoolingOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_pooling_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("idx");
}

const PbMessage& PoolingOp::GetSpecialConf() const {
  return op_conf().pooling_conf();
}

void PoolingOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  const Shape& input_shape = GetBlobDesc4BnInOp(SoleIbn())->shape();
  CHECK_EQ(input_shape.NumAxes(), 4);
  BlobDesc* output_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  const PoolingOpConf& pooling_conf = op_conf().pooling_conf();

  std::vector<int64_t> output_shape_dim_vec = {input_shape.At(0),
                                               input_shape.At(1)};

  output_shape_dim_vec.push_back((input_shape.At(2) + 2 * pooling_conf.pad_h()
                                  - pooling_conf.kernel_size_h())
                                     / pooling_conf.stride_h()
                                 + 1);

  output_shape_dim_vec.push_back((input_shape.At(3) + 2 * pooling_conf.pad_w()
                                  - pooling_conf.kernel_size_w())
                                     / pooling_conf.stride_w()
                                 + 1);

  output_blob_desc->mut_shape() = Shape(output_shape_dim_vec);
  BlobDesc* data_tmp_blob_desc = GetBlobDesc4BnInOp(SoleDtbn());
  data_tmp_blob_desc->mut_shape() = Shape(output_shape_dim_vec);
}

REGISTER_OP(OperatorConf::kPoolingConf, PoolingOp);

}  // namespace oneflow
