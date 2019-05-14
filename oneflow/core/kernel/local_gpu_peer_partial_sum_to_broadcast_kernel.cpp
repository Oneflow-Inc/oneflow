#include "oneflow/core/kernel/local_gpu_peer_partial_sum_to_broadcast_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

#ifdef WITH_CUDA

template<typename T>
void LocalGpuPeerPartialSumToBroadcastKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  Blob* buf = BnInOp2Blob("buf");
  CHECK_EQ(buf->data_type(), out->data_type());
  CHECK_EQ(buf->shape(), out->shape());
  FOR_RANGE(int32_t, i, 0, op_attribute().input_bns_size()) {
    const std::string& ibn = op_attribute().input_bns(i);
    CHECK_EQ(ibn, GenRepeatedBn("in", i));
    const Blob* in_i = BnInOp2Blob(ibn);
    CHECK_EQ(in_i->data_type(), out->data_type());
    CHECK_EQ(in_i->shape(), out->shape());
    if (i == 0) {
      CudaCheck(cudaMemcpyAsync(out->mut_dptr(), in_i->dptr(), out->ByteSizeOfDataContentField(),
                                cudaMemcpyDefault, ctx.device_ctx->cuda_stream()));
    } else {
      CudaCheck(cudaMemcpyAsync(buf->mut_dptr(), in_i->dptr(), out->ByteSizeOfDataContentField(),
                                cudaMemcpyDefault, ctx.device_ctx->cuda_stream()));
      Addition<DeviceType::kGPU, T>(ctx.device_ctx, out->shape().elem_cnt(), out->mut_dptr<T>(),
                                    out->dptr<T>(), in_i->dptr<T>());
    }
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalGpuPeerPartialSumToBroadcastConf,
                               LocalGpuPeerPartialSumToBroadcastKernel, ARITHMETIC_DATA_TYPE_SEQ)

#endif

}  // namespace oneflow
