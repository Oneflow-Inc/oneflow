#include "oneflow/core/operator/reduce_add_op.h"

namespace oneflow {

void ReduceAddOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_add_conf());
  for (int32_t i = 0; i < op_conf().reduce_add_conf().in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollDataTmpBn("copy_buf");
  EnrollOutputBn("out", false);
}

const PbMessage& ReduceAddOp::GetCustomizedConf() const { return op_conf().reduce_add_conf(); }

LogicalBlobId ReduceAddOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

void ReduceAddOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  int32_t in_num = op_conf().reduce_add_conf().in_num();
  CHECK_EQ(in_num, parallel_ctx->parallel_num());
  CHECK_GE(in_num, 2);
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  *GetBlobDesc4BnInOp("copy_buf") = *first_in_blob;
  *GetBlobDesc4BnInOp(SoleObn()) = *first_in_blob;
  for (int32_t i = 1; i < in_num; ++i) {
    CHECK(*first_in_blob == *GetBlobDesc4BnInOp(input_bns().Get(i)));
  }
}

REGISTER_OP(OperatorConf::kReduceAddConf, ReduceAddOp);

}  // namespace oneflow
