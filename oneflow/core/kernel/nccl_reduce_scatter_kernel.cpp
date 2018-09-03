#include "oneflow/core/kernel/nccl_reduce_scatter_kernel.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

// TODO(jiyuan): valid only in GPU device
template<DeviceType device_type>
void NcclReduceScatterKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  int32_t elem_cnt = out_blob->shape().elem_cnt();
  // CudaCheck(ncclGroupStart());
  CudaCheck(ncclReduceScatter((const void*)in_blob->dptr<>(), (void*)out_blob->mut_dptr<>(),
                              elem_cnt, ncclFloat, ncclSum, ctx.device_ctx->nccl_scatter_handle(),
                              ctx.device_ctx->cuda_stream()));
  //
  // CudaCheck(ncclGroupEnd());
  // cudaStreamSynchronize(ctx.device_ctx->cuda_stream());
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kNcclReduceScatterConf, NcclReduceScatterKernel);

}  // namespace oneflow
