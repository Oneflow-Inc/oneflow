#include "oneflow/core/operator/dot_op.h"
namespace oneflow {

void DotOp::InitFromOpConf() {
  CHECK(op_conf().has_dot_conf());

  EnrollInputBn("in");
  EnrollInputBn("weight");
  EnrollOutputBn("out");
  if (op_conf().dot_conf().has_bias()) { EnrollInputBn("bias"); }
}

const PbMessage& DotOp::GetCustomizedConf() const { return op_conf().dot_conf(); }

void DotOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext* parallel_ctx) const {
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  CHECK_EQ(in_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(in_blob_desc->shape().At(0), weight_blob_desc->shape().At(0));
  CHECK_EQ(in_blob_desc->shape().Count(1), weight_blob_desc->shape().Count(1));
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0)});
}

REGISTER_OP(OperatorConf::kDotConf, DotOp);

}  // namespace oneflow
