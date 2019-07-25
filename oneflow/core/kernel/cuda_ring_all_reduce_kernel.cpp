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
  arg.num_rings = 2;
  CHECK_GT(arg.num_rings, 0);
  CHECK_LE(arg.num_rings, CUDA_RING_ALL_REDUCE_MAX_NUM_RINGS);
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const int32_t num_step = conf.ring_conf(ring_id).step_conf_size();
  const Range range = Range(conf.ring_conf(ring_id).step_conf(step_id).data_range());

  arg.num_elem[0] = range.size() / 2;
  arg.num_elem[1] = range.size() / 2;

  Blob* send = BnInOp2Blob(GenRepeatedBn("send", ring_id));
  arg.send[0] = send != nullptr ? send->mut_dptr<T>() : nullptr;
  arg.send[1] = send != nullptr ? send->mut_dptr<T>() + arg.num_elem[0] : nullptr;
  const Blob* recv = BnInOp2Blob(GenRepeatedBn("recv", ring_id));
  arg.recv[0] = recv != nullptr ? recv->dptr<T>() : nullptr;
  arg.recv[1] = recv != nullptr ? recv->dptr<T>() + arg.num_elem[0] : nullptr;
  arg.src[0] = in != nullptr ? in->dptr<T>() + range.begin() : nullptr;
  arg.src[1] = in != nullptr ? in->dptr<T>() + range.begin() + arg.num_elem[0] : nullptr;
  arg.dst[0] = out != nullptr ? out->mut_dptr<T>() + range.begin() : nullptr;
  arg.dst[1] = out != nullptr ? out->mut_dptr<T>() + range.begin() + arg.num_elem[0] : nullptr;

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
