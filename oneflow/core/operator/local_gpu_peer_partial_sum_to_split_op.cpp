#include "oneflow/core/operator/local_gpu_peer_partial_sum_to_split_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void LocalGpuPeerPartialSumToSplitOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in", false);
  EnrollOutputBn("out", false);
  EnrollFwBufBn("buf");
}

const PbMessage& LocalGpuPeerPartialSumToSplitOp::GetCustomizedConf() const {
  return op_conf().local_gpu_peer_partial_sum_to_split_conf();
}

void LocalGpuPeerPartialSumToSplitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const LocalGpuPeerPartialSumToSplitOpConf& conf =
      op_conf().local_gpu_peer_partial_sum_to_split_conf();
  const int64_t out_split_axis = conf.out_split_axis();
  const BlobDesc* in_0 = GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  const int64_t num_axes = in_0->shape().NumAxes();
  CHECK_GE(out_split_axis, 0);
  CHECK_LT(out_split_axis, num_axes);
  FOR_RANGE(int64_t, i, 1, input_bns().size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    CHECK_EQ(in_i->data_type(), in_0->data_type());
    CHECK_EQ(in_i->shape().NumAxes(), num_axes);
    FOR_RANGE(int64_t, axis, 0, in_i->shape().NumAxes()) {
      CHECK_EQ(in_0->shape().At(axis), in_i->shape().At(axis));
    }
  }
  const int64_t split_dim_size = in_0->shape().At(out_split_axis);
  const BalancedSplitter out_split_bs(split_dim_size, parallel_ctx->parallel_num());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in_0;
  out->mut_shape().Set(out_split_axis, out_split_bs.At(parallel_ctx->parallel_id()).size());
  *GetBlobDesc4BnInOp("buf") = *out;
}

LogicalNode* LocalGpuPeerPartialSumToSplitOp::NewProperLogicalNode() const {
  return new LocalGpuPeerBoxingLogicalNode();
}

void LocalGpuPeerPartialSumToSplitOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  LocalGpuPeerPartialSumToSplitKernelConf* conf =
      kernel_conf->mutable_local_gpu_peer_partial_sum_to_split_conf();
  const int64_t out_split_axis =
      op_conf().local_gpu_peer_partial_sum_to_split_conf().out_split_axis();
  const BalancedSplitter bs(GetBlobDesc4BnInOp("in")->shape().At(out_split_axis),
                            parallel_ctx->parallel_num());
  conf->mutable_range()->set_begin(bs.At(parallel_ctx->parallel_id()).begin());
  conf->mutable_range()->set_end(bs.At(parallel_ctx->parallel_id()).end());
}

REGISTER_OP(OperatorConf::kLocalGpuPeerPartialSumToSplitConf, LocalGpuPeerPartialSumToSplitOp);

}  // namespace oneflow
