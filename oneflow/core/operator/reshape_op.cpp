#include "oneflow/core/operator/reshape_op.h"

namespace oneflow {

void ReshapeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ReshapeOp::GetCustomizedConf() const { return op_conf().reshape_conf(); }

void ReshapeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;

  CHECK(!in_blob_desc->has_instance_shape_field());
  const ReshapeOpConf& conf = op_conf().reshape_conf();
  std::vector<int64_t> dim_vec(1 + conf.shape().dim_size());
  dim_vec[0] = in_blob_desc->shape().At(0);
  FOR_RANGE(size_t, i, 1, dim_vec.size()) { dim_vec[i] = conf.shape().dim(i - 1); }
  out_blob_desc->mut_shape() = Shape(dim_vec);
  CHECK_EQ(out_blob_desc->shape().elem_cnt(), in_blob_desc->shape().elem_cnt());
}

REGISTER_OP(OperatorConf::kReshapeConf, ReshapeOp);

}  // namespace oneflow
