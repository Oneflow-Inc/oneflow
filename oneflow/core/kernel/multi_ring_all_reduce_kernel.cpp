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

inline int64_t KernelCtxGetStepId(const KernelCtx& ctx) {
  return static_cast<std::pair<int64_t, int64_t>*>(ctx.other)->second;
}

}  // namespace

template<DeviceType device_type, typename T>
void MultiRingAllReduceKernel<device_type, T>::VirtualKernelInit(const ParallelContext* ctx) {
  const MultiRingAllReduceOpConf& conf = this->op_conf().multi_ring_all_reduce_conf();
  FOR_RANGE(int64_t, ring_id, 0, conf.rings_size()) {
    send_bn_.push_back(GenRepeatedBn("send", ring_id));
    recv_bn_.push_back(GenRepeatedBn("recv", ring_id));
  }
}

template<DeviceType device_type, typename T>
void MultiRingAllReduceKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const MultiRingAllReduceKernelConf& all_reduce_kernel_conf =
      this->kernel_conf().multi_ring_all_reduce_conf();
  const int64_t ring_id = KernelCtxGetRingId(ctx);
  const int64_t step_id = KernelCtxGetStepId(ctx);
  const MultiRingAllReduceKernelStepConf& step_conf =
      all_reduce_kernel_conf.ring_conf(ring_id).step_conf(step_id);
  const Blob* in = BnInOp2Blob("in");
  const Blob* recv = BnInOp2Blob(recv_bn_.at(ring_id));
  Blob* out = BnInOp2Blob("out");
  Blob* send = BnInOp2Blob(send_bn_.at(ring_id));
  using Util = MultiRingAllReduceKernelUtil<device_type, T>;
  const int64_t step_offset = step_conf.data_range().begin();
  const int64_t step_size = step_conf.data_range().end() - step_offset;
  if (step_conf.send() && !step_conf.recv() && !step_conf.reduce() && !step_conf.copy()) {
    Util::Copy(ctx.device_ctx, send->mut_dptr<T>(), in->dptr<T>() + step_offset, step_size);
  } else if (step_conf.send() && step_conf.recv() && step_conf.reduce() && !step_conf.copy()) {
    Util::Reduce(ctx.device_ctx, send->mut_dptr<T>(), recv->dptr<T>(), in->dptr<T>() + step_offset,
                 step_size);
  } else if (step_conf.send() && step_conf.recv() && step_conf.reduce() && step_conf.copy()) {
    Util::Reduce(ctx.device_ctx, out->mut_dptr<T>() + step_offset, send->mut_dptr<T>(),
                 recv->dptr<T>(), in->dptr<T>() + step_offset, step_size);
  } else if (step_conf.send() && step_conf.recv() && !step_conf.reduce() && step_conf.copy()) {
    Util::Copy(ctx.device_ctx, send->mut_dptr<T>(), out->mut_dptr<T>() + step_offset,
               recv->dptr<T>(), step_size);
  } else if (!step_conf.send() && step_conf.recv() && !step_conf.reduce() && step_conf.copy()) {
    Util::Copy(ctx.device_ctx, out->mut_dptr<T>() + step_offset, recv->dptr<T>(), step_size);
  } else {
    UNIMPLEMENTED();
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMultiRingAllReduceConf, MultiRingAllReduceKernel,
                           FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
