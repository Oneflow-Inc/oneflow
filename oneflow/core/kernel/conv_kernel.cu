#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void CudnnConvKernel<T>::VirtualKernelInit(const ParallelContext*) {
  const int kernel_dim_size = static_cast<int>(
      this->GetInt32FromCustomizedKernelConf("kernel_dim_size"));

  Shape in_shape(this->kernel_conf().conv_conf().in());
  Shape out_shape(this->kernel_conf().conv_conf().out());
  Shape weight_shape(this->kernel_conf().conv_conf().weight());

  // TODO()
  in_desc_ = new CudnnTensorDesc(CudnnDataType<T>::val, in_shape);
  out_desc_ = new CudnnTensorDesc(CudnnDataType<T>::val, out_shape);
  filter_desc_ =
      new CudnnFilterDesc(CudnnDataType<T>::val, weight_shape,
                          this->GetStringFromCustomizedOpConf("data_format"));
  conv_desc_ = new CudnnConvDesc(
      CudnnDataType<T>::val,
      this->GetInt32FromCustomizedKernelConf("kernel_dim_size"),
      this->GetPbRfFromCustomizedOpConf("dilation_rate").data(),
      this->GetPbRfFromCustomizedOpConf("strides").data(),
      this->GetPbRfFromCustomizedOpConf("kernel_size").data(),
      this->GetStringFromCustomizedOpConf("data_format"),
      this->GetStringFromCustomizedOpConf("padding"));

  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    Shape bias_shape(this->kernel_conf().conv_conf().bias());
    this->bias_desc_ = new CudnnTensorDesc(CudnnDataType<T>::val, bias_shape);
  }
}

template<typename T>
void CudnnConvKernel<T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* cudnn_workspace = BnInOp2Blob("cudnn_workspace");

  cudnnConvolutionFwdAlgo_t cudnn_fwd_algo =
      static_cast<cudnnConvolutionFwdAlgo_t>(
          kernel_conf().conv_conf().cudnn_fwd_algo());
  CudaCheck(cudnnConvolutionForward(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one, in_desc_->Get(),
      in_blob->dptr<T>(), filter_desc_->Get(), weight_blob->dptr<T>(),
      conv_desc_->Get(), cudnn_fwd_algo, cudnn_workspace->mut_dptr<T>(),
      cudnn_workspace->shape().At(0), CudnnDataType<T>::zero, out_desc_->Get(),
      out_blob->mut_dptr<T>()));

  if (this->GetBoolFromCustomizedOpConf<T>("use_bias")) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(ctx.device_ctx->cudnn_handle(),
                             CudnnDataType<T>::one, bias_desc_->Get(),
                             bias_blob->dptr<T>(), CudnnDataType<T>::one,
                             out_desc_->Get(), out_blob->mut_dptr<T>()));
  }
}

template<typename T>
void CudnnConvKernel<T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");

  // compute bias_diff
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    Blob* bias_diff_blob = BnInOp2Blob("bias_diff");
    Memset<DeviceType::kGPU>(ctx.device_ctx, bias_diff_blob->mut_dptr(), 0,
                             bias_diff_blob->ByteSizeOfDataContentField());

    CudaCheck(cudnnConvolutionBackwardBias(
        ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one, out_desc_->Get(),
        out_diff_blob->dptr<T>(), CudnnDataType<T>::one, bias_desc_->Get(),
        bias_diff_blob->mut_dptr<T>()));
  }

  // compute weight_diff
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  Blob* cudnn_workspace = BnInOp2Blob("cudnn_workspace");
  Memset<DeviceType::kGPU>(ctx.device_ctx, weight_diff_blob->mut_dptr(), 0,
                           weight_diff_blob->ByteSizeOfDataContentField());

  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo =
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          kernel_conf().conv_conf().cudnn_bwd_filter_algo());

  CudaCheck(cudnnConvolutionBackwardFilter(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one, in_desc_->Get(),
      in_blob->dptr<T>(), out_desc_->Get(), out_diff_blob->dptr<T>(),
      conv_desc_->Get(), cudnn_bwd_filter_algo, cudnn_workspace->mut_dptr<T>(),
      cudnn_workspace->shape().At(0), CudnnDataType<T>::one,
      filter_desc_->Get(), weight_diff_blob->mut_dptr<T>()));

  // compute in_diff
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());

  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
      static_cast<cudnnConvolutionBwdDataAlgo_t>(
          kernel_conf().conv_conf().cudnn_bwd_data_algo());

  CudaCheck(cudnnConvolutionBackwardData(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one,
      filter_desc_->Get(), weight_blob->dptr<T>(), out_desc_->Get(),
      out_diff_blob->dptr<T>(), conv_desc_->Get(), cudnn_bwd_data_algo,
      cudnn_workspace->mut_dptr<T>(), cudnn_workspace->shape().At(0),
      CudnnDataType<T>::zero, in_desc_->Get(), in_diff_blob->mut_dptr<T>()));
}

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class CudnnConvKernel<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
