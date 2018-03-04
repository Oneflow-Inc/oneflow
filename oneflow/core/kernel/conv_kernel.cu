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

  std::vector<int32_t> stride_of_in_tensor(kernel_dim_size, 1);
  std::vector<int32_t> stride_of_out_tensor(kernel_dim_size, 1);
  for (int32_t i = kernel_dim_size + 2 - 1; i > 0; --i) {
    for (int32_t j = kernel_dim_size + 2 - 2; j >= 0; --j) {
      stride_of_in_tensor[j] *= in_shape.At(i);
      stride_of_out_tensor[j] *= out_shape.At(i);
    }
  }
  std::vector<int32_t> in_dim(in_shape.dim_vec().begin(),
                              in_shape.dim_vec().end());
  std::vector<int32_t> out_dim(out_shape.dim_vec().begin(),
                               out_shape.dim_vec().end());

  this->in_desc_.reset(new CudnnTensorDesc(GetDataType<T>::val,
                                           kernel_dim_size + 2, in_dim.data(),
                                           stride_of_in_tensor.data()));
  this->out_desc_.reset(new CudnnTensorDesc(GetDataType<T>::val,
                                            kernel_dim_size + 2, out_dim.data(),
                                            stride_of_out_tensor.data()));
  this->filter_desc_.reset(
      new CudnnFilterDesc(GetDataType<T>::val, weight_shape,
                          this->GetStringFromCustomizedOpConf("data_format")));
  this->conv_desc_.reset(new CudnnConvDesc(
      GetDataType<T>::val, in_shape,
      this->GetInt32FromCustomizedKernelConf("kernel_dim_size"),
      this->GetInt32PbRfFromCustomizedOpConf("dilation_rate").data(),
      this->GetInt32PbRfFromCustomizedOpConf("strides").data(),
      this->GetInt32PbRfFromCustomizedOpConf("kernel_size").data(),
      this->GetStringFromCustomizedOpConf("data_format"),
      this->GetStringFromCustomizedOpConf("padding")));

  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    int32_t filters = this->GetInt32FromCustomizedOpConf("filters");
    int32_t stride_of_bias_tensor = 1;
    this->bias_desc_.reset(new CudnnTensorDesc(GetDataType<T>::val, 1, &filters,
                                               &stride_of_bias_tensor));
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
          this->kernel_conf().conv_conf().cudnn_fwd_algo());
  CudaCheck(cudnnConvolutionForward(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one,
      this->in_desc_->Get(), in_blob->dptr<T>(), this->filter_desc_->Get(),
      weight_blob->dptr<T>(), this->conv_desc_->Get(), cudnn_fwd_algo,
      cudnn_workspace->mut_dptr<T>(), cudnn_workspace->shape().At(0),
      CudnnDataType<T>::zero, this->out_desc_->Get(), out_blob->mut_dptr<T>()));

  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(ctx.device_ctx->cudnn_handle(),
                             CudnnDataType<T>::one, this->bias_desc_->Get(),
                             bias_blob->dptr<T>(), CudnnDataType<T>::one,
                             this->out_desc_->Get(), out_blob->mut_dptr<T>()));
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
        ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one,
        this->out_desc_->Get(), out_diff_blob->dptr<T>(), CudnnDataType<T>::one,
        this->bias_desc_->Get(), bias_diff_blob->mut_dptr<T>()));
  }

  // compute weight_diff
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  Blob* cudnn_workspace = BnInOp2Blob("cudnn_workspace");
  Memset<DeviceType::kGPU>(ctx.device_ctx, weight_diff_blob->mut_dptr(), 0,
                           weight_diff_blob->ByteSizeOfDataContentField());

  cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo =
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->kernel_conf().conv_conf().cudnn_bwd_filter_algo());

  CudaCheck(cudnnConvolutionBackwardFilter(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one,
      this->in_desc_->Get(), in_blob->dptr<T>(), this->out_desc_->Get(),
      out_diff_blob->dptr<T>(), this->conv_desc_->Get(), cudnn_bwd_filter_algo,
      cudnn_workspace->mut_dptr<T>(), cudnn_workspace->shape().At(0),
      CudnnDataType<T>::one, this->filter_desc_->Get(),
      weight_diff_blob->mut_dptr<T>()));

  // compute in_diff
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());

  cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo =
      static_cast<cudnnConvolutionBwdDataAlgo_t>(
          this->kernel_conf().conv_conf().cudnn_bwd_data_algo());

  CudaCheck(cudnnConvolutionBackwardData(
      ctx.device_ctx->cudnn_handle(), CudnnDataType<T>::one,
      this->filter_desc_->Get(), weight_blob->dptr<T>(), this->out_desc_->Get(),
      out_diff_blob->dptr<T>(), this->conv_desc_->Get(), cudnn_bwd_data_algo,
      cudnn_workspace->mut_dptr<T>(), cudnn_workspace->shape().At(0),
      CudnnDataType<T>::zero, this->in_desc_->Get(),
      in_diff_blob->mut_dptr<T>()));
}

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class CudnnConvKernel<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
