#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::PoolingForward(const KernelCtx& kernel_ctx,
                                                        const PoolingCtx& pooling_ctx,
                                                        const Blob* in_blob, Blob* out_blob) const {
  CudaCheck(cudnnPoolingForward(
      kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), CudnnSPOnePtr<T>(),
      pooling_ctx.cudnn_in_tensor_desc(), in_blob->dptr(), CudnnSPZeroPtr<T>(),
      pooling_ctx.cudnn_out_tensor_desc(), out_blob->mut_dptr()));
}

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::PoolingBackward(const KernelCtx& kernel_ctx,
                                                         const PoolingCtx& pooling_ctx,
                                                         const Blob* out_diff_blob,
                                                         const Blob* out_blob, const Blob* in_blob,
                                                         Blob* in_diff_blob) const {
  CudaCheck(cudnnPoolingBackward(
      kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), CudnnSPOnePtr<T>(),
      pooling_ctx.cudnn_out_tensor_desc(), out_blob->dptr(), pooling_ctx.cudnn_out_tensor_desc(),
      out_diff_blob->dptr(), pooling_ctx.cudnn_in_tensor_desc(), in_blob->dptr(),
      CudnnSPZeroPtr<T>(), pooling_ctx.cudnn_in_tensor_desc(), in_diff_blob->mut_dptr()));
}

#define INSTANTIATE_POOLING_KERNEL(type_cpp, type_proto) \
  template class PoolingKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
