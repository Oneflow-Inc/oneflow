#include "oneflow/core/operator/local_gpu_peer_split_to_split_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void LocalGpuPeerSplitToSplitOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& LocalGpuPeerSplitToSplitOp::GetCustomizedConf() const {
  return op_conf().local_gpu_peer_split_to_split_conf();
}

void LocalGpuPeerSplitToSplitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const LocalGpuPeerSplitToSplitOpConf& conf = op_conf().local_gpu_peer_split_to_split_conf();
  const int64_t in_split_axis = conf.in_split_axis();
  const int64_t out_split_axis = conf.out_split_axis();
  const BlobDesc* in_0 = GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  const int64_t num_axes = in_0->shape().NumAxes();
  CHECK_GE(in_split_axis, 0);
  CHECK_LT(in_split_axis, num_axes);
  CHECK_GE(out_split_axis, 0);
  CHECK_LT(out_split_axis, num_axes);
  std::vector<int64_t> parts;
  parts.push_back(in_0->shape().At(in_split_axis));
  FOR_RANGE(int64_t, i, 1, input_bns().size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    CHECK_EQ(in_i->data_type(), in_0->data_type());
    CHECK_EQ(in_i->shape().NumAxes(), num_axes);
    FOR_RANGE(int64_t, axis, 0, in_i->shape().NumAxes()) {
      if (axis == in_split_axis) {
        parts.push_back(in_i->shape().At(axis));
      } else {
        CHECK_EQ(in_0->shape().At(axis), in_i->shape().At(axis));
      }
    }
  }
  const int64_t in_split_dim_size = std::accumulate(parts.cbegin(), parts.cend(), 0L);
  const BalancedSplitter in_bs(in_split_dim_size, conf.in_size());
  FOR_RANGE(int64_t, i, 0, conf.in_size()) { CHECK_EQ(in_bs.At(i).size(), parts.at(i)); }
  const int64_t split_dim_size = in_0->shape().At(out_split_axis);
  const BalancedSplitter out_split_bs(split_dim_size, parallel_ctx->parallel_num());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in_0;
  out->mut_shape().Set(in_split_axis, in_split_dim_size);
  out->mut_shape().Set(out_split_axis, out_split_bs.At(parallel_ctx->parallel_id()).size());
}

LogicalNode* LocalGpuPeerSplitToSplitOp::NewProperLogicalNode() const {
  return new LocalGpuPeerBoxingLogicalNode();
}

REGISTER_OP(OperatorConf::kLocalGpuPeerSplitToSplitConf, LocalGpuPeerSplitToSplitOp);

}  // namespace oneflow
