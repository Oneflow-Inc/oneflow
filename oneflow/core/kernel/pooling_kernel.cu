#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::ForwardDataContent(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const Pooling3DCtx& pooling_ctx = this->pooling_3d_ctx();
  T alpha = 1.0;
  T beta = 0.0;
  CudaCheck(cudnnPoolingForward(
      kernel_ctx.device_ctx->cudnn_handle(),
      pooling_ctx.pooling_desc_ptr()->Get(), &alpha,
      pooling_ctx.in_desc_ptr()->Get(), in_blob->dptr(), &beta,
      pooling_ctx.out_desc_ptr()->Get(), out_blob->mut_dptr()));
}

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::BackwardDataContent(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kGPU>(kernel_ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  const Pooling3DCtx& pooling_ctx = this->pooling_3d_ctx();
  T alpha = 1.0;
  T beta = 0.0;
  CudaCheck(cudnnPoolingBackward(
      kernel_ctx.device_ctx->cudnn_handle(),
      pooling_ctx.pooling_desc_ptr()->Get(), &alpha,
      pooling_ctx.out_desc_ptr()->Get(), out_blob->dptr(),
      pooling_ctx.out_desc_ptr()->Get(), out_diff_blob->dptr(),
      pooling_ctx.in_desc_ptr()->Get(), in_blob->dptr(), &beta,
      pooling_ctx.in_desc_ptr()->Get(), in_diff_blob->mut_dptr()));
}

#define INSTANTIATE_POOLING_KERNEL(type_cpp, type_proto) \
  template class PoolingKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
