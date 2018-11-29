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

  const ReshapeOpConf conf = op_conf().reshape_conf();
  ReshapeType reshape_type;
  reshape_type.set_has_dim0_in_shape(conf.has_dim0_in_shape());
  reshape_type.mutable_shape()->CopyFrom(conf.shape());

  out_blob_desc->mut_shape() = GetShapeFromReshapeTypeConf(reshape_type, in_blob_desc->shape());
  ;
}

REGISTER_OP(OperatorConf::kReshapeConf, ReshapeOp);

}  // namespace oneflow
