#include "oneflow/core/operator/reduce_gather_op.h"

namespace oneflow {

void ReduceGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_gather_conf());
  for (int32_t i = 0; i < op_conf().reduce_gather_conf().in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("out", false);
}

const PbMessage& ReduceGatherOp::GetCustomizedConf() const {
  return op_conf().reduce_gather_conf();
}

void ReduceGatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  int32_t in_num = op_conf().reduce_gather_conf().in_num();
  CHECK_EQ(in_num, parallel_ctx->parallel_num());
  CHECK_GE(in_num, 2);
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *first_in_blob;
  int64_t out_blob_elem_cnt = first_in_blob->shape().elem_cnt();
  for (int32_t i = 1; i < in_num; ++i) {
    out_blob_elem_cnt += GetBlobDesc4BnInOp(input_bns().Get(i))->shape().elem_cnt();
  }
  out_blob->mut_shape() = Shape({out_blob_elem_cnt});
}

REGISTER_OP(OperatorConf::kReduceGatherConf, ReduceGatherOp);

}  // namespace oneflow
