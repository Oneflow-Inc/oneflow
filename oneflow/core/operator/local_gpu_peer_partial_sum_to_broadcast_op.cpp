#include "oneflow/core/operator/local_gpu_peer_partial_sum_to_broadcast_op.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void LocalGpuPeerPartialSumToBroadcastOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in", false);
  EnrollOutputBn("out", false);
  EnrollFwBufBn("buf");
}

const PbMessage& LocalGpuPeerPartialSumToBroadcastOp::GetCustomizedConf() const {
  return op_conf().local_gpu_peer_partial_sum_to_broadcast_conf();
}

void LocalGpuPeerPartialSumToBroadcastOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_0 = GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  const int64_t num_axes = in_0->shape().NumAxes();
  FOR_RANGE(int64_t, i, 1, input_bns().size()) {
    const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    CHECK_EQ(in_i->data_type(), in_0->data_type());
    CHECK_EQ(in_i->shape().NumAxes(), num_axes);
    FOR_RANGE(int64_t, axis, 0, in_i->shape().NumAxes()) {
      CHECK_EQ(in_0->shape().At(axis), in_i->shape().At(axis));
    }
  }
  *GetBlobDesc4BnInOp("out") = *in_0;
  *GetBlobDesc4BnInOp("buf") = *in_0;
}

LogicalNode* LocalGpuPeerPartialSumToBroadcastOp::NewProperLogicalNode() const {
  return new LocalGpuPeerBoxingLogicalNode();
}

REGISTER_OP(OperatorConf::kLocalGpuPeerPartialSumToBroadcastConf,
            LocalGpuPeerPartialSumToBroadcastOp);

}  // namespace oneflow
