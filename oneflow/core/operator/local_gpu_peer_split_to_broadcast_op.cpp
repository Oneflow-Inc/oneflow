#include "oneflow/core/operator/local_gpu_peer_split_to_broadcast_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void LocalGpuPeerSplitToBroadcastOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& LocalGpuPeerSplitToBroadcastOp::GetCustomizedConf() const {
  return op_conf().local_gpu_peer_split_to_broadcast_conf();
}

void LocalGpuPeerSplitToBroadcastOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const LocalGpuPeerSplitToBroadcastOpConf& conf =
      op_conf().local_gpu_peer_split_to_broadcast_conf();
  const int64_t in_split_axis = conf.in_split_axis();
  const BlobDesc* in_0 = GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  const int64_t num_axes = in_0->shape().NumAxes();
  CHECK_GE(in_split_axis, 0);
  CHECK_LT(in_split_axis, num_axes);
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
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in_0;
  out->mut_shape().Set(in_split_axis, in_split_dim_size);
}

LogicalNode* LocalGpuPeerSplitToBroadcastOp::NewProperLogicalNode() const {
  return new LocalGpuPeerBoxingLogicalNode();
}

void LocalGpuPeerSplitToBroadcastOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  LocalGpuPeerSplitToBroadcastKernelConf* conf =
      kernel_conf->mutable_local_gpu_peer_split_to_broadcast_conf();
  const int64_t in_split_axis = op_conf().local_gpu_peer_split_to_broadcast_conf().in_split_axis();
  int64_t start = 0;
  FOR_RANGE(int64_t, i, 0, input_bns().size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    const int64_t size = in_i->shape().At(in_split_axis);
    RangeProto* range = conf->mutable_in_split()->mutable_range()->Add();
    range->set_start(start);
    range->set_end(start + size);
    start = range->end();
  }
}

REGISTER_OP(OperatorConf::kLocalGpuPeerSplitToBroadcastConf, LocalGpuPeerSplitToBroadcastOp);

}  // namespace oneflow
