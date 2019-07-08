#include "oneflow/core/operator/multi_ring_all_reduce_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void MultiRingAllReduceOp::InitFromOpConf() {
  const MultiRingAllReduceOpConf& conf = op_conf().multi_ring_all_reduce_conf();
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
  const PbRpf<RingConf>& rings = conf.rings();
  CHECK_GE(rings.size(), 1);
  const int64_t num_rank = rings.Get(0).next_size();
  FOR_RANGE(int64_t, i, 1, rings.size()) { CHECK_EQ(rings.Get(i).next_size(), num_rank); }
  EnrollRepeatedInputBn("recv", conf.rings_size(), false);
  EnrollRepeatedOutputBn("send", conf.rings_size(), false);
}

LogicalBlobId MultiRingAllReduceOp::ibn2lbi(const std::string& input_bn) const {
  return op_conf().multi_ring_all_reduce_conf().lbi();
}

LogicalBlobId MultiRingAllReduceOp::obn2lbi(const std::string& output_bn) const {
  return op_conf().multi_ring_all_reduce_conf().lbi();
}

const PbMessage& MultiRingAllReduceOp::GetCustomizedConf() const {
  return op_conf().multi_ring_all_reduce_conf();
}

void MultiRingAllReduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const MultiRingAllReduceOpConf& conf = op_conf().multi_ring_all_reduce_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in->shape(), Shape(conf.logical_blob_shape()));
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  const BalancedSplitter range4ring(in->shape().elem_cnt(), conf.rings_size());
  FOR_RANGE(int64_t, i, 0, conf.rings_size()) {
    const int64_t ring_elem_cnt = range4ring.At(i).size();
    const int64_t buffer_size =
        RoundUp(ring_elem_cnt, conf.rings(i).next_size()) / conf.rings(i).next_size();
    BlobDesc* send_i = GetBlobDesc4BnInOp(GenRepeatedBn("send", i));
    send_i->set_data_type(in->data_type());
    send_i->mut_shape() = Shape({buffer_size});
  }
}

void MultiRingAllReduceOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  const MultiRingAllReduceOpConf& conf = this->op_conf().multi_ring_all_reduce_conf();
  MultiRingAllReduceKernelConf* multi_ring_all_reduce_kernel_conf =
      kernel_conf->mutable_multi_ring_all_reduce_conf();
  multi_ring_all_reduce_kernel_conf->set_num_rank(conf.rings(0).next_size());
  const BalancedSplitter ring2range(GetBlobDesc4BnInOp("in")->shape().elem_cnt(),
                                    conf.rings_size());
  FOR_RANGE(int64_t, ring_id, 0, conf.rings_size()) {
    const Range ring_range = ring2range.At(ring_id);
    const RingConf& ring = conf.rings(ring_id);
    CHECK_EQ(ring.next_size(), parallel_ctx->parallel_num());
    const int64_t num_steps = ring.next_size() * 2 - 1;
    std::vector<int64_t> ring_prev(ring.next_size());
    FOR_RANGE(int64_t, i, 0, ring.next_size()) { ring_prev[ring.next(i)] = i; }
    int64_t current_slice_id = ring_prev[parallel_ctx->parallel_id()];
    const BalancedSplitter slices(ring_range.size(), ring.next_size());
    std::vector<Range> chunks;
    MultiRingAllReduceKernelRingConf* kernel_ring_conf =
        multi_ring_all_reduce_kernel_conf->mutable_ring_conf()->Add();
    FOR_RANGE(int64_t, i, 0, num_steps) {
      MultiRingAllReduceKernelStepConf* step_conf = kernel_ring_conf->mutable_step_conf()->Add();
      step_conf->mutable_data_range()->set_begin(slices.At(current_slice_id).begin()
                                                 + ring_range.begin());
      step_conf->mutable_data_range()->set_end(slices.At(current_slice_id).end()
                                               + ring_range.begin());
      if (i == 0) {
        step_conf->set_send(true);
        step_conf->set_recv(false);
        step_conf->set_reduce(false);
        step_conf->set_copy(false);
      } else if (i < ring.next_size() - 1) {
        step_conf->set_send(true);
        step_conf->set_recv(true);
        step_conf->set_reduce(true);
        step_conf->set_copy(false);
      } else if (i == ring.next_size() - 1) {
        step_conf->set_send(true);
        step_conf->set_recv(true);
        step_conf->set_reduce(true);
        step_conf->set_copy(true);
      } else if (i < num_steps - 1) {
        step_conf->set_send(true);
        step_conf->set_recv(true);
        step_conf->set_reduce(false);
        step_conf->set_copy(true);
      } else if (i == num_steps - 1) {
        step_conf->set_send(false);
        step_conf->set_recv(true);
        step_conf->set_reduce(false);
        step_conf->set_copy(true);
      } else {
        UNIMPLEMENTED();
      }
      current_slice_id = ring_prev[current_slice_id];
    }
  }
}

REGISTER_OP(OperatorConf::kMultiRingAllReduceConf, MultiRingAllReduceOp);

}  // namespace oneflow
