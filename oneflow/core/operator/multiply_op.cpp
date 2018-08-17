#include "oneflow/core/operator/multiply_op.h"
#include "oneflow/core/common/balanced_splitter.h"
namespace oneflow {

void MultiplyOp::InitFromOpConf() {
  CHECK(op_conf().has_multiply_conf());
  EnrollInputBn("in");
  EnrollInputBn("weight");
  EnrollOutputBn("out");
}

const PbMessage& MultiplyOp::GetCustomizedConf() const { return op_conf().multiply_conf(); }

void MultiplyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  CHECK_EQ(in_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(in_blob_desc->shape(), weight_blob_desc->shape());
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
}
REGISTER_OP(OperatorConf::kMultiplyConf, MultiplyOp);

}  // namespace oneflow
