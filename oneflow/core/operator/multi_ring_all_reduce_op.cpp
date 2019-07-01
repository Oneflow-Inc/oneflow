#include "oneflow/core/operator/multi_ring_all_reduce_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void MultiRingAllReduceOp::InitFromOpConf() {
  const MultiRingAllReduceOpConf& conf = op_conf().multi_ring_all_reduce_conf();
  EnrollInputBn("in", false);
  EnrollOutputBn("send", false);
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
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  const BalancedSplitter range4ring(in->shape().elem_cnt(), conf.rings_size());
  FOR_RANGE(int64_t, i, 0, conf.rings_size()) {
    const int64_t ring_elem_cnt = range4ring.At(i).size();
    const int64_t buffer_size = RoundUp(ring_elem_cnt, conf.rings(i).next_size());
    BlobDesc* send_i = GetBlobDesc4BnInOp(GenRepeatedBn("send", i));
    send_i->set_data_type(in->data_type());
    send_i->mut_shape() = Shape({buffer_size});
    if (ring_elem_cnt % buffer_size != 0) {
      send_i->set_has_dim0_valid_num_field(true);
      send_i->mut_dim0_inner_shape() = Shape({1, buffer_size});
    }
  }
}

REGISTER_OP(OperatorConf::kMultiRingAllReduceConf, MultiRingAllReduceOp);

}  // namespace oneflow
