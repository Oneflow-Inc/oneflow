#include "oneflow/core/operator/batch_gather_op.h"

namespace oneflow {

void BatchGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_batch_gather_conf());
  EnrollInputBn("in");
  EnrollInputBn("indices", false);
  EnrollOutputBn("out");
}

const PbMessage& BatchGatherOp::GetCustomizedConf() const { return op_conf().batch_gather_conf(); }

void BatchGatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_GT(in_blob_desc->shape().NumAxes(), 0);
  const BlobDesc* indices_blob_desc = GetBlobDesc4BnInOp("indices");
  CHECK_GT(indices_blob_desc->shape().NumAxes(), 0);
  CHECK(IsIntegralDataType(indices_blob_desc->data_type()));
  const std::vector<int64_t>& in_dim_vec = in_blob_desc->shape().dim_vec();
  const std::vector<int64_t>& indices_dim_vec = indices_blob_desc->shape().dim_vec();
  CHECK_LE(indices_dim_vec.size(), in_dim_vec.size());

  // out
  std::vector<int64_t> out_dim_vec;
  for (int32_t i = 0; i < indices_dim_vec.size() - 1; i++) {
    CHECK_EQ(indices_dim_vec.at(i), in_dim_vec.at(i));
    out_dim_vec.push_back(indices_dim_vec.at(i));
  }
  out_dim_vec.push_back(indices_dim_vec.back());
  for (int32_t i = indices_dim_vec.size(); i < in_dim_vec.size(); i++) {
    out_dim_vec.push_back(in_dim_vec.at(i));
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
}

REGISTER_OP(OperatorConf::kBatchGatherConf, BatchGatherOp);

}  // namespace oneflow