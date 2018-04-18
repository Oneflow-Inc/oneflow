#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::VirtualKernelInit(
    const ParallelContext* parallel_ctx) {
  Shape in_shape(this->GetConvKernelConf().in());
  Shape out_shape(this->GetConvKernelConf().out());
  Shape weight_shape(this->GetConvKernelConf().weight());

  this->in_desc_.reset(new CudnnTensorDesc(
      GetDataType<T>::value, in_shape,
      this->template GetValFromCustomizedOpConf<std::string>("data_format")));
  this->out_desc_.reset(new CudnnTensorDesc(
      GetDataType<T>::value, out_shape,
      this->template GetValFromCustomizedOpConf<std::string>("data_format")));
  this->filter_desc_.reset(new CudnnFilterDesc(
      GetDataType<T>::value, weight_shape,
      this->template GetValFromCustomizedOpConf<std::string>("data_format")));
  this->conv_desc_.reset(new CudnnConvDesc(GetDataType<T>::value, in_shape,
                                           this->GetCustomizedOpConf()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    int32_t filters =
        this->template GetValFromCustomizedOpConf<int32_t>("filters");

    if (this->OpKernelDim() == 2) {
      if (this->template GetValFromCustomizedOpConf<std::string>("data_format")
          == "channels_first") {
        this->bias_desc_.reset(new CudnnTensorDesc(
            CUDNN_TENSOR_NCHW, GetDataType<T>::value, 1, filters, 1, 1));
      } else if (this->template GetValFromCustomizedOpConf<std::string>(
                     "data_format")
                 == "channels_last") {
        this->bias_desc_.reset(new CudnnTensorDesc(
            CUDNN_TENSOR_NHWC, GetDataType<T>::value, 1, filters, 1, 1));
      } else {
        UNIMPLEMENTED();
      }
    } else {
      std::vector<int32_t> bias_dim(this->OpKernelDim() + 2, 1);
      std::vector<int32_t> stride_of_bias_tensor(this->OpKernelDim() + 2, 1);
      bias_dim[1] = filters;
      stride_of_bias_tensor[0] = filters;
      this->bias_desc_.reset(
          new CudnnTensorDesc(GetDataType<T>::value, this->OpKernelDim() + 2,
                              bias_dim.data(), stride_of_bias_tensor.data()));
    }
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::DoForwardDataContent(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob,
    Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* cudnn_buf = BnInOp2Blob("cudnn_buf");
  int64_t cudnn_buf_size = 0;
  T* cudnn_buf_dptr = nullptr;
  if (cudnn_buf) {
    cudnn_buf_size = cudnn_buf->shape().At(0);
    cudnn_buf_dptr = cudnn_buf->mut_dptr<T>();
  }
  CudaCheck(cudnnConvolutionForward(
      device_ctx->cudnn_handle(), OnePtr<T>::value, this->in_desc_->Get(),
      in_blob->dptr<T>(), this->filter_desc_->Get(), weight_blob->dptr<T>(),
      this->conv_desc_->Get(),
      static_cast<cudnnConvolutionFwdAlgo_t>(
          this->GetConvKernelConf().cudnn_fwd_algo()),
      cudnn_buf_dptr, cudnn_buf_size, ZeroPtr<T>::value, this->out_desc_->Get(),
      out_blob->mut_dptr<T>()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    const Blob* bias = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(device_ctx->cudnn_handle(), OnePtr<T>::value,
                             this->bias_desc_->Get(), bias->dptr<T>(),
                             OnePtr<T>::value, this->out_desc_->Get(),
                             out_blob->mut_dptr<T>()));
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::WeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* in_blob,
    Blob* weight_diff_blob, Blob* in_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* cudnn_buf = BnInOp2Blob("cudnn_buf");
  int64_t cudnn_buf_size = 0;
  T* cudnn_buf_dptr = nullptr;
  if (cudnn_buf) {
    cudnn_buf_size = cudnn_buf->shape().At(0);
    cudnn_buf_dptr = cudnn_buf->mut_dptr<T>();
  }
  CudaCheck(cudnnConvolutionBackwardFilter(
      device_ctx->cudnn_handle(), OnePtr<T>::value, this->in_desc_->Get(),
      in_blob->dptr<T>(), this->out_desc_->Get(), out_diff_blob->dptr<T>(),
      this->conv_desc_->Get(),
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->GetConvKernelConf().cudnn_bwd_filter_algo()),
      cudnn_buf_dptr, cudnn_buf_size, ZeroPtr<T>::value,
      this->filter_desc_->Get(), weight_diff_blob->mut_dptr<T>()));

  if (in_diff_blob != nullptr) {
    CudaCheck(cudnnConvolutionBackwardData(
        device_ctx->cudnn_handle(), OnePtr<T>::value, this->filter_desc_->Get(),
        weight_blob->dptr<T>(), this->out_desc_->Get(),
        out_diff_blob->dptr<T>(), this->conv_desc_->Get(),
        static_cast<cudnnConvolutionBwdDataAlgo_t>(
            this->GetConvKernelConf().cudnn_bwd_data_algo()),
        cudnn_buf_dptr, cudnn_buf_size, ZeroPtr<T>::value,
        this->in_desc_->Get(), in_diff_blob->mut_dptr<T>()));
  }
}

template<typename T>
void ConvKernel<DeviceType::kGPU, T>::BiasBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, Blob* bias_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CudaCheck(cudnnConvolutionBackwardBias(
      device_ctx->cudnn_handle(), OnePtr<T>::value, this->out_desc_->Get(),
      out_diff_blob->dptr<T>(), ZeroPtr<T>::value, this->bias_desc_->Get(),
      bias_diff_blob->mut_dptr<T>()));
}

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class ConvKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
