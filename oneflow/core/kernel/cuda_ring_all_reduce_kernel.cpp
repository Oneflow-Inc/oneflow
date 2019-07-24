#include "oneflow/core/kernel/cuda_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void CudaRingAllReduceKernel<T>::VirtualKernelInit(const ParallelContext* ctx) {}

template<typename T>
void CudaRingAllReduceKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t step = *static_cast<int64_t*>(ctx.other);
  const MultiRingAllReduceKernelConf& conf = kernel_conf().multi_ring_all_reduce_conf();
  CudaRingAllReduceArg<T> arg{};
  arg.num_rings = conf.ring_conf_size();
  CHECK_GT(arg.num_rings, 0);
  CHECK_LE(arg.num_rings, CUDA_RING_ALL_REDUCE_MAX_NUM_RINGS);
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const int32_t num_step = conf.ring_conf(0).step_conf_size();
  FOR_RANGE(int32_t, i, 0, arg.num_rings) {
    Blob* send = BnInOp2Blob(GenRepeatedBn("send", i));
    arg.send[i] = send != nullptr ? send->mut_dptr<T>() : nullptr;
    const Blob* recv = BnInOp2Blob(GenRepeatedBn("recv", i));
    arg.recv[i] = recv != nullptr ? recv->dptr<T>() : nullptr;
    const Range range = Range(conf.ring_conf(i).step_conf(step).data_range());
    arg.src[i] = in != nullptr ? in->dptr<T>() + range.begin() : nullptr;
    arg.dst[i] = out != nullptr ? out->mut_dptr<T>() + range.begin() : nullptr;
    arg.num_elem[i] = range.size();
  }
  if (step == 0) {
    CudaRingAllReduceKernelUtil<T>::Send(ctx.device_ctx, arg);
  } else if (step < conf.num_rank() - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvReduceSend(ctx.device_ctx, arg);
  } else if (step == conf.num_rank() - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvReduceSendCopy(ctx.device_ctx, arg);
  } else if (step < num_step - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvSendCopy(ctx.device_ctx, arg);
  } else if (step == num_step - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvCopy(ctx.device_ctx, arg);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCudaRingAllReduceConf, CudaRingAllReduceKernel,
                               FLOATING_DATA_TYPE_SEQ)
}  // namespace oneflow
