#include "oneflow/core/kernel/cuda_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void CudaRingAllReduceKernel<T>::VirtualKernelInit(const ParallelContext* ctx) {}

template<typename T>
void CudaRingAllReduceKernel<T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto other_ctx = *static_cast<std::pair<int64_t, int64_t>*>(ctx.other);
  const int64_t ring_id = other_ctx.first;
  const int64_t step_id = other_ctx.second;
  const MultiRingAllReduceKernelConf& conf = kernel_conf().multi_ring_all_reduce_conf();
  CudaRingAllReduceParams<T> params{};
  params.num_links = 1;
  CHECK_GT(params.num_links, 0);
  CHECK_LE(params.num_links, CUDA_RING_ALL_REDUCE_MAX_NUM_LINK);
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const int32_t num_step = conf.ring_conf(ring_id).step_conf_size();
  const Range range = Range(conf.ring_conf(ring_id).step_conf(step_id).data_range());
  params.links[0].num_elem = range.size() / 2;
  Blob* send = BnInOp2Blob(GenRepeatedBn("send", ring_id));
  params.links[0].send = send != nullptr ? send->mut_dptr<T>() : nullptr;
  const Blob* recv = BnInOp2Blob(GenRepeatedBn("recv", ring_id));
  params.links[0].recv = recv != nullptr ? recv->dptr<T>() : nullptr;
  params.links[0].src = in != nullptr ? in->dptr<T>() + range.begin() : nullptr;
  params.links[0].dst = out != nullptr ? out->mut_dptr<T>() + range.begin() : nullptr;
  if (step_id == 0) {
    CudaRingAllReduceKernelUtil<T>::Send(ctx.device_ctx, params);
  } else if (step_id < conf.num_rank() - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvReduceSend(ctx.device_ctx, params);
  } else if (step_id == conf.num_rank() - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvReduceSendCopy(ctx.device_ctx, params);
  } else if (step_id < num_step - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvSendCopy(ctx.device_ctx, params);
  } else if (step_id == num_step - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvCopy(ctx.device_ctx, params);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCudaRingAllReduceConf, CudaRingAllReduceKernel,
                               FLOATING_DATA_TYPE_SEQ)
}  // namespace oneflow
