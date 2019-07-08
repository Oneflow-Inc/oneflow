#include "oneflow/core/kernel/multi_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/multi_ring_all_reduce_kernel_util.h"

namespace oneflow {

namespace {

inline int64_t KernelCtxGetRingId(const KernelCtx& ctx) {
  return static_cast<std::pair<int64_t, int64_t>*>(ctx.other)->first;
}

inline int64_t KernelCtxGetStep(const KernelCtx& ctx) {
  return static_cast<std::pair<int64_t, int64_t>*>(ctx.other)->second;
}

}  // namespace

template<DeviceType device_type, typename T>
void MultiRingAllReduceKernel<device_type, T>::VirtualKernelInit(const ParallelContext* ctx) {
  const MultiRingAllReduceOpConf& conf = this->op_conf().multi_ring_all_reduce_conf();
  const BalancedSplitter ring2range(Shape(conf.logical_blob_shape()).elem_cnt(), conf.rings_size());
  FOR_RANGE(int64_t, ring_id, 0, conf.rings_size()) {
    const Range ring_range = ring2range.At(ring_id);
    const RingConf& ring = conf.rings(ring_id);
    CHECK_EQ(ring.next_size(), ctx->parallel_num());
    const int64_t num_steps = ring.next_size() * 2 - 1;
    if (num_steps_ == -1) {
      num_steps_ = num_steps;
    } else {
      CHECK_EQ(num_steps_, num_steps);
    }
    std::vector<int64_t> ring_prev(ring.next_size());
    FOR_RANGE(int64_t, i, 0, ring.next_size()) { ring_prev[ring.next(i)] = i; }
    int64_t current_slice_id = ring_prev[ctx->parallel_id()];
    const BalancedSplitter slices(ring_range.size(), ring.next_size());
    std::vector<Range> chunks;
    FOR_RANGE(int64_t, i, 0, num_steps_) {
      chunks.emplace_back(slices.At(current_slice_id).begin() + ring_range.begin(),
                          slices.At(current_slice_id).end() + ring_range.begin());
      current_slice_id = ring_prev[current_slice_id];
    }
    chunk_slices_.push_back(chunks);
    send_bn_.push_back(GenRepeatedBn("send", ring_id));
    recv_bn_.push_back(GenRepeatedBn("recv", ring_id));
  }
}

template<DeviceType device_type, typename T>
void MultiRingAllReduceKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const MultiRingAllReduceOpConf& conf = this->op_conf().multi_ring_all_reduce_conf();
  const int64_t ring_id = KernelCtxGetRingId(ctx);
  const int64_t step = KernelCtxGetStep(ctx);
  const Blob* in = BnInOp2Blob("in");
  const Blob* recv = BnInOp2Blob(recv_bn_.at(ring_id));
  Blob* out = BnInOp2Blob("out");
  Blob* send = BnInOp2Blob(send_bn_.at(ring_id));
  const Range& range = chunk_slices_.at(ring_id).at(step);
  const RingConf& ring = conf.rings(ring_id);
  using Util = MultiRingAllReduceKernelUtil<device_type, T>;
  if (step == 0) {
    Util::Copy(ctx.device_ctx, send->mut_dptr<T>(), in->dptr<T>() + range.begin(), range.size());
  } else if (step < ring.next_size() - 1) {
    Util::Reduce(ctx.device_ctx, send->mut_dptr<T>(), recv->dptr<T>(),
                 in->dptr<T>() + range.begin(), range.size());
  } else if (step == ring.next_size() - 1) {
    Util::Reduce(ctx.device_ctx, out->mut_dptr<T>() + range.begin(), send->mut_dptr<T>(),
                 recv->dptr<T>(), in->dptr<T>() + range.begin(), range.size());

  } else if (step < num_steps_ - 1) {
    Util::Copy(ctx.device_ctx, send->mut_dptr<T>(), out->mut_dptr<T>() + range.begin(),
               recv->dptr<T>(), range.size());
  } else if (step == num_steps_ - 1) {
    Util::Copy(ctx.device_ctx, out->mut_dptr<T>() + range.begin(), recv->dptr<T>(), range.size());
  } else {
    UNIMPLEMENTED();
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMultiRingAllReduceConf, MultiRingAllReduceKernel,
                           FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
