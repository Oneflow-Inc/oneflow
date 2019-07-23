#include "oneflow/core/kernel/cuda_ring_all_reduce_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void CudaRingAllReduceKernel<T>::VirtualKernelInit(const ParallelContext* ctx) {}

template<typename T>
void CudaRingAllReduceKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const CudaRingAllReduceOpConf& conf = this->op_conf().cuda_ring_all_reduce_conf();
  CudaRingAllReduceArg<T> arg{};
  arg.num_rings = conf.rings_size();
  CHECK_LE(arg.num_rings, CUDA_RING_ALL_REDUCE_MAX_NUM_RINGS);
  FOR_RANGE(int32_t, i, 0, arg.num_rings) {
    arg.send[i] = BnInOp2Blob(GenRepeatedBn("send", i))->mut_dptr<T>();
    arg.recv[i] = BnInOp2Blob(GenRepeatedBn("recv", i))->mut_dptr<T>();
  }
  CudaRingAllReduceKernelUtil<T>::AllReduce(ctx.device_ctx, arg);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCudaRingAllReduceConf, CudaRingAllReduceKernel,
                               FLOATING_DATA_TYPE_SEQ)
}  // namespace oneflow
