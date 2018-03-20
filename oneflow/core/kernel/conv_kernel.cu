#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::VirtualKernelInit(
    const ParallelContext* parallel_ctx) {
  Shape in_shape(static_cast<const ShapeProto&>(
      this->GetMessageFromCustomizedKernelConf("in")));
  Shape out_shape(static_cast<const ShapeProto&>(
      this->GetMessageFromCustomizedKernelConf("out")));
  Shape weight_shape(static_cast<const ShapeProto&>(
      this->GetMessageFromCustomizedKernelConf("weight")));

  std::vector<int32_t> stride_of_in_tensor(this->KernelDim(), 1);
  std::vector<int32_t> stride_of_out_tensor(this->KernelDim(), 1);
  for (int32_t i = this->KernelDim() + 2 - 1; i > 0; --i) {
    for (int32_t j = this->KernelDim() + 2 - 2; j >= 0; --j) {
      stride_of_in_tensor[j] *= in_shape.At(i);
      stride_of_out_tensor[j] *= out_shape.At(i);
    }
  }
  std::vector<int32_t> in_dim(in_shape.dim_vec().begin(),
                              in_shape.dim_vec().end());
  std::vector<int32_t> out_dim(out_shape.dim_vec().begin(),
                               out_shape.dim_vec().end());

  this->in_desc_.reset(new CudnnTensorDesc(GetDataType<T>::val,
                                           this->KernelDim() + 2, in_dim.data(),
                                           stride_of_in_tensor.data()));
  this->out_desc_.reset(
      new CudnnTensorDesc(GetDataType<T>::val, this->KernelDim() + 2,
                          out_dim.data(), stride_of_out_tensor.data()));
  this->filter_desc_.reset(
      new CudnnFilterDesc(GetDataType<T>::val, weight_shape,
                          this->GetStringFromCustomizedOpConf("data_format")));
  this->conv_desc_.reset(new CudnnConvDesc(GetDataType<T>::val, in_shape,
                                           this->GetCustomizedOpConf()));

  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    int32_t filters = this->GetInt32FromCustomizedOpConf("filters");
    int32_t stride_of_bias_tensor = 1;
    this->bias_desc_.reset(new CudnnTensorDesc(GetDataType<T>::val, 1, &filters,
                                               &stride_of_bias_tensor));
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::WeightForward(
    DeviceCtx* device_ctx, const Blob* in, const Blob* weight, Blob* out,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* cudnn_buf = BnInOp2Blob("cudnn_buf");
  CudaCheck(cudnnConvolutionForward(
      device_ctx->cudnn_handle(), CudnnDataType<T>::one, this->in_desc_->Get(),
      in->dptr<T>(), this->filter_desc_->Get(), weight->dptr<T>(),
      this->conv_desc_->Get(),
      static_cast<cudnnConvolutionFwdAlgo_t>(
          this->GetInt32FromCustomizedKernelConf("cudnn_fwd_algo")),
      cudnn_buf->mut_dptr<T>(), cudnn_buf->shape().At(0),
      CudnnDataType<T>::zero, this->out_desc_->Get(), out->mut_dptr<T>()));
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::BiasForward(DeviceCtx* device_ctx,
                                                  const Blob* bias,
                                                  Blob* out) const {
  CudaCheck(cudnnAddTensor(device_ctx->cudnn_handle(), CudnnDataType<T>::one,
                           this->bias_desc_->Get(), bias->dptr<T>(),
                           CudnnDataType<T>::one, this->out_desc_->Get(),
                           out->mut_dptr<T>()));
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::DataBackward(
    DeviceCtx* device_ctx, const Blob* out_diff, const Blob* weight,
    Blob* in_diff, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* cudnn_buf = BnInOp2Blob("cudnn_buf");
  CudaCheck(cudnnConvolutionBackwardData(
      device_ctx->cudnn_handle(), CudnnDataType<T>::one,
      this->filter_desc_->Get(), weight->dptr<T>(), this->out_desc_->Get(),
      out_diff->dptr<T>(), this->conv_desc_->Get(),
      static_cast<cudnnConvolutionBwdDataAlgo_t>(
          this->GetInt32FromCustomizedKernelConf("cudnn_bwd_data_algo")),
      cudnn_buf->mut_dptr<T>(), cudnn_buf->shape().At(0),
      CudnnDataType<T>::zero, this->in_desc_->Get(), in_diff->mut_dptr<T>()));
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::WeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff, const Blob* in,
    Blob* weight_diff,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* cudnn_buf = BnInOp2Blob("cudnn_buf");
  CudaCheck(cudnnConvolutionBackwardFilter(
      device_ctx->cudnn_handle(), CudnnDataType<T>::one, this->in_desc_->Get(),
      in->dptr<T>(), this->out_desc_->Get(), out_diff->dptr<T>(),
      this->conv_desc_->Get(),
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->GetInt32FromCustomizedKernelConf("cudnn_bwd_filter_algo")),
      cudnn_buf->mut_dptr<T>(), cudnn_buf->shape().At(0), CudnnDataType<T>::one,
      this->filter_desc_->Get(), weight_diff->mut_dptr<T>()));
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::BiasBackward(DeviceCtx* device_ctx,
                                                   const Blob* out_diff,
                                                   Blob* bias_diff) const {
  CudaCheck(cudnnConvolutionBackwardBias(
      device_ctx->cudnn_handle(), CudnnDataType<T>::one, this->out_desc_->Get(),
      out_diff->dptr<T>(), CudnnDataType<T>::one, this->bias_desc_->Get(),
      bias_diff->mut_dptr<T>()));
}

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class ConvKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
