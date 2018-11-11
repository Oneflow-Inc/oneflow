#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("indices");
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetCustomizedConf() const { return op_conf().gather_conf(); }

void GatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices_blob_desc = GetBlobDesc4BnInOp("indices");
  CHECK_EQ(indices_blob_desc->data_type(), DataType::kInt32);
  CHECK_GE(indices_blob_desc->shape().NumAxes(), 1);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_GE(in_blob_desc->shape().NumAxes(), 1);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *indices_blob_desc;
  out_blob_desc->set_data_type(in_blob_desc->data_type());
  std::vector<int64_t> out_shape_dim_vec = indices_blob_desc->shape().dim_vec();
  out_shape_dim_vec.insert(out_shape_dim_vec.end(), in_blob_desc->shape().dim_vec().cbegin() + 1,
                           in_blob_desc->shape().dim_vec().cend());
  out_blob_desc->mut_shape() = Shape(out_shape_dim_vec);
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
