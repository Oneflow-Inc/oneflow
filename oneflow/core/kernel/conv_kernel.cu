#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void CudnnConvKernel<T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto conv_conf = this->op_conf().convolution_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  const Blob* bias_blob = BnInOp2Blob("bias");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* cudnn_workspace = BnInOp2Blob("cudnn_workspace");

  this->cudnn_conv_desc_.InitFromBlobDescAndOpConf(
      in_blob->blob_desc_ptr(), out_blob->blob_desc_ptr(), conv_conf);

  this->cudnn_conv_desc_.Forward<T>(
      ctx.device_ctx->cudnn_handle(), in_blob, weight_blob, bias_blob, out_blob,
      cudnn_workspace,
      static_cast<cudnnConvolutionFwdAlgo_t>(
          this->kernel_conf().convolution_conf().cudnn_fwd_algo()),
      conv_conf.has_bias_term());
}

template<typename T>
void CudnnConvKernel<T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto conv_conf = this->op_conf().convolution_conf();
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");

  // compute bias_diff
  if (conv_conf.has_bias_term()) {
    Blob* bias_diff_blob = BnInOp2Blob("bias_diff");
    Memset<DeviceType::kGPU>(ctx.device_ctx, bias_diff_blob->mut_dptr(), 0,
                             bias_diff_blob->ByteSizeOfDataContentField());
    this->cudnn_conv_desc_.BackwardBias<T>(ctx.device_ctx->cudnn_handle(),
                                           out_diff_blob, bias_diff_blob);
  }

  // compute weight_diff
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  Blob* cudnn_workspace = BnInOp2Blob("cudnn_workspace");
  Memset<DeviceType::kGPU>(ctx.device_ctx, weight_diff_blob->mut_dptr(), 0,
                           weight_diff_blob->ByteSizeOfDataContentField());
  this->cudnn_conv_desc_.BackwardFilter<T>(
      ctx.device_ctx->cudnn_handle(), in_blob, out_diff_blob, weight_diff_blob,
      cudnn_workspace,
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->kernel_conf().convolution_conf().cudnn_bwd_filter_algo()));

  // compute in_diff
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());
  this->cudnn_conv_desc_.BackwardData<T>(
      ctx.device_ctx->cudnn_handle(), weight_blob, out_diff_blob, in_diff_blob,
      cudnn_workspace,
      static_cast<cudnnConvolutionBwdDataAlgo_t>(
          this->kernel_conf().convolution_conf().cudnn_bwd_data_algo()));
}

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class CudnnConvKernel<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
