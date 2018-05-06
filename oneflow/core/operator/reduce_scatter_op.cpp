#include "oneflow/core/operator/reduce_scatter_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ReduceScatterOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_scatter_conf());
  EnrollInputBn("in", false);
  for (int32_t i = 0; i < op_conf().reduce_scatter_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& ReduceScatterOp::GetCustomizedConf() const {
  return op_conf().reduce_scatter_conf();
}

LogicalBlobId ReduceScatterOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

void ReduceScatterOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                                     const ParallelContext* parallel_ctx) const {
  int32_t out_num = op_conf().reduce_scatter_conf().out_num();
  BlobDesc* in_blob = GetBlobDesc4BnInOp(SoleIbn());
  BalancedSplitter splitter(in_blob->shape().elem_cnt(), out_num);
  CHECK_EQ(out_num, parallel_ctx->parallel_num());
  CHECK_GE(out_num, 2);
  for (int32_t i = 0; i < out_num; ++i) {
    BlobDesc* out_blob = GetBlobDesc4BnInOp(output_bns().Get(i));
    *out_blob = *in_blob;
    out_blob->mut_shape() = Shape({splitter.At(i).size()});
  }
}

REGISTER_OP(OperatorConf::kReduceScatterConf, ReduceScatterOp);

}  // namespace oneflow
