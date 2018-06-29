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

void ReduceScatterOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t out_num = op_conf().reduce_scatter_conf().out_num();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp(SoleIbn());
  BalancedSplitter model_splitter(in_blob->shape().elem_cnt(), parallel_ctx->parallel_num());
  BalancedSplitter device_splitter(parallel_ctx->parallel_num(), out_num);
  CHECK_GE(out_num, 2);
  for (int32_t i = 0; i < out_num; ++i) {
    BlobDesc* out_blob = GetBlobDesc4BnInOp(output_bns().Get(i));
    *out_blob = *in_blob;
    Range model_range = device_splitter.At(i);
    out_blob->mut_shape() =
        Shape({model_splitter.At(model_range.begin(), model_range.end() - 1).size()});
  }
}

REGISTER_OP(OperatorConf::kReduceScatterConf, ReduceScatterOp);

}  // namespace oneflow
