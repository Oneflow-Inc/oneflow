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
  CudaRingAllReduceArg<T> arg{};
  arg.num_rings = conf.ring_conf_size();
  CHECK_GT(arg.num_rings, 0);
  CHECK_LE(arg.num_rings, CUDA_RING_ALL_REDUCE_MAX_NUM_RINGS);
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const int32_t num_step = conf.ring_conf(ring_id).step_conf_size();

  Blob* send = BnInOp2Blob(GenRepeatedBn("send", ring_id));
  arg.send[ring_id] = send != nullptr ? send->mut_dptr<T>() : nullptr;
  const Blob* recv = BnInOp2Blob(GenRepeatedBn("recv", ring_id));
  arg.recv[ring_id] = recv != nullptr ? recv->dptr<T>() : nullptr;
  const Range range = Range(conf.ring_conf(ring_id).step_conf(step_id).data_range());
  arg.src[ring_id] = in != nullptr ? in->dptr<T>() + range.begin() : nullptr;
  arg.dst[ring_id] = out != nullptr ? out->mut_dptr<T>() + range.begin() : nullptr;
  arg.num_elem[ring_id] = range.size();

  if (step_id == 0) {
    CudaRingAllReduceKernelUtil<T>::Send(ctx.device_ctx, arg);
  } else if (step_id < conf.num_rank() - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvReduceSend(ctx.device_ctx, arg);
  } else if (step_id == conf.num_rank() - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvReduceSendCopy(ctx.device_ctx, arg);
  } else if (step_id < num_step - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvSendCopy(ctx.device_ctx, arg);
  } else if (step_id == num_step - 1) {
    CudaRingAllReduceKernelUtil<T>::RecvCopy(ctx.device_ctx, arg);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCudaRingAllReduceConf, CudaRingAllReduceKernel,
                               FLOATING_DATA_TYPE_SEQ)
}  // namespace oneflow
