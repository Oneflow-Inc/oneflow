#include "oneflow/core/kernel/cuda_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/cuda_ring_boxing_kernel_util.h"

namespace oneflow {

template<typename T>
void CudaRingAllReduceKernel<T>::VirtualKernelInit(const ParallelContext* ctx) {}

template<typename T>
void CudaRingAllReduceKernel<T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto other_ctx = *static_cast<std::pair<int64_t, int64_t>*>(ctx.other);
  const int64_t step_id = other_ctx.first;
  const int64_t link_dup_id = other_ctx.second;
  const CudaRingAllReduceKernelConf& conf = kernel_conf().cuda_ring_all_reduce_conf();
  CudaRingBoxingStepParams<T> params{};
  const CudaRingAllReduceStepConf& step_conf = conf.step_conf(step_id);
  params.num_links = step_conf.link_conf(link_dup_id).link_data_range_size();
  CHECK_GT(params.num_links, 0);
  CHECK_LE(params.num_links, CUDA_RING_BOXING_MAX_NUM_LINK);
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  params.recv = step_conf.recv();
  params.in = step_conf.in();
  params.send = step_conf.send();
  params.out = step_conf.out();
  FOR_RANGE(int64_t, link_id, 0, params.num_links) {
    const Range range = Range(step_conf.link_conf(link_dup_id).link_data_range(link_id));
    params.links[link_id].num_elem = range.size();
    Blob* send = BnInOp2Blob(GenRepeatedBn("send", link_id));
    params.links[link_id].send = send != nullptr ? send->mut_dptr<T>() : nullptr;
    const Blob* recv = BnInOp2Blob(GenRepeatedBn("recv", link_id));
    params.links[link_id].recv = recv != nullptr ? recv->dptr<T>() : nullptr;
    params.links[link_id].in = in != nullptr ? in->dptr<T>() + range.begin() : nullptr;
    params.links[link_id].out = out != nullptr ? out->mut_dptr<T>() + range.begin() : nullptr;
  }
  CudaRingBoxingKernelUtil<ReduceMethod::kSum, T>::LaunchGenericRingStep(ctx.device_ctx, params);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCudaRingAllReduceConf, CudaRingAllReduceKernel,
                               FLOATING_DATA_TYPE_SEQ)
}  // namespace oneflow
