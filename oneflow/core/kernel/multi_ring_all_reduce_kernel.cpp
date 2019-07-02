#include "oneflow/core/kernel/multi_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

int64_t KernelCtxGetRingId(const KernelCtx& ctx) {
  return static_cast<std::pair<int64_t, int64_t>*>(ctx.other)->first;
}

int64_t KernelCtxGetStep(const KernelCtx& ctx) {
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
    std::vector<int64_t> ring_prev(conf.rings_size());
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
  }
}

template<DeviceType device_type, typename T>
void MultiRingAllReduceKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t ring_id = KernelCtxGetRingId(ctx);
  const int64_t step = KernelCtxGetStep(ctx);
  if (step == 0) {
    BnInOp2Blob("send")->set_dim0_valid_num(0, chunk_slices_.at(ring_id).at(step).size());
  } else if (step < num_steps_) {
    BnInOp2Blob("send")->set_dim0_valid_num(0, BnInOp2Blob("recv")->dim0_valid_num(0));
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void MultiRingAllReduceKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const MultiRingAllReduceOpConf& conf = this->op_conf().multi_ring_all_reduce_conf();
  const int64_t ring_id = KernelCtxGetRingId(ctx);
  const int64_t step = KernelCtxGetStep(ctx);
  const Blob* in = BnInOp2Blob("in");
  const Blob* recv = BnInOp2Blob(GenRepeatedBn("recv", ring_id));
  Blob* out = BnInOp2Blob("out");
  Blob* send = BnInOp2Blob(GenRepeatedBn("send", ring_id));
  const Range& range = chunk_slices_.at(ring_id).at(step);
  const RingConf& ring = conf.rings(ring_id);
  if (step == 0) {
    Memcpy<device_type>(ctx.device_ctx, send->mut_dptr<T>(), in->dptr<T>() + range.begin(),
                        range.size() * sizeof(T));
  } else if (step < ring.next_size() - 1) {
    Addition<device_type, T>(ctx.device_ctx, range.size(), send->mut_dptr<T>(), recv->dptr<T>(),
                             in->dptr<T>() + range.begin());
  } else if (step == ring.next_size() - 1) {
    Addition<device_type, T>(ctx.device_ctx, range.size(), send->mut_dptr<T>(), recv->dptr<T>(),
                             in->dptr<T>() + range.begin());
    Memcpy<device_type>(ctx.device_ctx, out->mut_dptr<T>() + range.begin(), send->dptr<T>(),
                        range.size() * sizeof(T));
  } else if (step < num_steps_ - 1) {
    Memcpy<device_type>(ctx.device_ctx, out->mut_dptr<T>() + range.begin(), recv->dptr<T>(),
                        range.size() * sizeof(T));
    Memcpy<device_type>(ctx.device_ctx, send->mut_dptr<T>(), recv->dptr<T>(),
                        range.size() * sizeof(T));
  } else if (step == num_steps_ - 1) {
    Memcpy<device_type>(ctx.device_ctx, out->mut_dptr<T>() + range.begin(), recv->dptr<T>(),
                        range.size() * sizeof(T));
  } else {
    UNIMPLEMENTED();
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMultiRingAllReduceConf, MultiRingAllReduceKernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
