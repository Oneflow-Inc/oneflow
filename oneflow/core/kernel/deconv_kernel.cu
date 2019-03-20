#include "oneflow/core/kernel/deconv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
void DeconvKernel<DeviceType::kGPU, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  Shape in_shape(this->GetDeconvKernelConf().in());
  Shape out_shape(this->GetDeconvKernelConf().out());
  Shape weight_shape(this->GetDeconvKernelConf().weight());

  const std::string& data_format =
      this->template GetValFromCustomizedOpConf<std::string>("data_format");
  this->in_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, in_shape, data_format));
  this->out_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, out_shape, data_format));
  this->filter_desc_.reset(new CudnnFilterDesc(GetDataType<T>::value, weight_shape, data_format));
  this->deconv_desc_.reset(
      new CudnnDeconvDesc(GetDataType<T>::value, in_shape, this->GetCustomizedOpConf()));

  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    int32_t filters = this->template GetValFromCustomizedOpConf<int32_t>("filters");
    if (this->OpKernelDim() == 2) {
      if (data_format == "channels_first") {
        this->bias_desc_.reset(
            new CudnnTensorDesc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, 1, filters, 1, 1));
      } else if (data_format == "channels_last") {
        if (GetDataType<T>::value == DataType::kDouble) {
          LOG(FATAL) << "CUDNN 1d & 2d support channels last only if data type "
                        "is float";
        }
        this->bias_desc_.reset(
            new CudnnTensorDesc(CUDNN_TENSOR_NHWC, GetDataType<T>::value, 1, filters, 1, 1));
      } else {
        UNIMPLEMENTED();
      }
    } else {
      if (data_format == "channels_last") {
        LOG(FATAL) << "CUDNN Nd API only support channels first";
      }
      std::vector<int32_t> bias_dim(this->OpKernelDim() + 2, 1);
      std::vector<int32_t> stride_of_bias_tensor(this->OpKernelDim() + 2, 1);
      bias_dim[1] = filters;
      stride_of_bias_tensor[0] = filters;
      this->bias_desc_.reset(new CudnnTensorDesc(GetDataType<T>::value, this->OpKernelDim() + 2,
                                                 bias_dim.data(), stride_of_bias_tensor.data()));
    }
  }
}

template<typename T>
void DeconvKernel<DeviceType::kGPU, T>::UpdateCudnnDescIfNeed(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // if (!(this->kernel_conf().need_do_instance_shape())) { return; }
  CHECK(this->EnableCudnn());

  Blob* out_or_diff_blob = BnInOp2Blob("out");
  if (!out_or_diff_blob) {
    out_or_diff_blob = BnInOp2Blob(GenDiffBn("out"));
    CHECK(out_or_diff_blob);
  }

  const std::string& data_format =
      this->template GetValFromCustomizedOpConf<std::string>("data_format");
  this->in_desc_.reset(
      new CudnnTensorDesc(GetDataType<T>::value, BnInOp2Blob("in")->shape(), data_format));
  this->out_desc_.reset(
      new CudnnTensorDesc(GetDataType<T>::value, out_or_diff_blob->shape(), data_format));
  this->deconv_desc_.reset(new CudnnDeconvDesc(GetDataType<T>::value, BnInOp2Blob("in")->shape(),
                                               this->GetCustomizedOpConf()));
}

template<typename T>
void DeconvKernel<DeviceType::kGPU, T>::DoForwardDataContent(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* bw_cudnn_buf = BnInOp2Blob("bw_cudnn_buf");
  void* bw_cudnn_buf_ptr = bw_cudnn_buf ? bw_cudnn_buf->mut_dptr() : nullptr;
  size_t bw_cudnn_buf_size = bw_cudnn_buf ? bw_cudnn_buf->ByteSizeOfDataContentField() : 0;
  //  Filters
  CudaCheck(cudnnConvolutionBackwardData(
      device_ctx->cudnn_handle(), OnePtr<T>::value, this->filter_desc_->Get(),
      weight_blob->dptr<T>(), this->in_desc_->Get(), in_blob->dptr<T>(), this->deconv_desc_->Get(),
      static_cast<cudnnConvolutionBwdDataAlgo_t>(this->GetDeconvKernelConf().cudnn_bwd_data_algo()),
      bw_cudnn_buf_ptr, bw_cudnn_buf_size, ZeroPtr<T>::value, this->out_desc_->Get(),
      out_blob->mut_dptr<T>()));
  //  Bias
  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    const Blob* bias = BnInOp2Blob("bias");
    CudaCheck(cudnnAddTensor(device_ctx->cudnn_handle(), OnePtr<T>::value, this->bias_desc_->Get(),
                             bias->dptr<T>(), OnePtr<T>::value, this->out_desc_->Get(),
                             out_blob->mut_dptr<T>()));
  }
}

template<typename T>
void DeconvKernel<DeviceType::kGPU, T>::WeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* in_blob, Blob* weight_diff_blob,
    Blob* in_diff_blob, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* fw_cudnn_buf = BnInOp2Blob("fw_cudnn_buf");
  void* fw_cudnn_buf_ptr = fw_cudnn_buf ? fw_cudnn_buf->mut_dptr() : nullptr;
  size_t fw_cudnn_buf_size = fw_cudnn_buf ? fw_cudnn_buf->ByteSizeOfDataContentField() : 0;
  Blob* bw_cudnn_buf = BnInOp2Blob("bw_cudnn_buf");
  void* bw_cudnn_buf_ptr = bw_cudnn_buf ? bw_cudnn_buf->mut_dptr() : nullptr;
  size_t bw_cudnn_buf_size = bw_cudnn_buf ? bw_cudnn_buf->ByteSizeOfDataContentField() : 0;

  CudaCheck(cudnnConvolutionBackwardFilter(
      device_ctx->cudnn_handle(), OnePtr<T>::value, this->out_desc_->Get(),
      out_diff_blob->dptr<T>(), this->in_desc_->Get(), in_blob->dptr<T>(),
      this->deconv_desc_->Get(),
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          this->GetDeconvKernelConf().cudnn_bwd_filter_algo()),
      bw_cudnn_buf_ptr, bw_cudnn_buf_size, ZeroPtr<T>::value, this->filter_desc_->Get(),
      weight_diff_blob->mut_dptr<T>()));

  if (in_diff_blob != nullptr) {
    CudaCheck(cudnnConvolutionForward(
        device_ctx->cudnn_handle(), OnePtr<T>::value, this->out_desc_->Get(),
        out_diff_blob->dptr<T>(), this->filter_desc_->Get(), weight_blob->dptr<T>(),
        this->deconv_desc_->Get(),
        static_cast<cudnnConvolutionFwdAlgo_t>(this->GetDeconvKernelConf().cudnn_fwd_algo()),
        fw_cudnn_buf_ptr, fw_cudnn_buf_size, ZeroPtr<T>::value, this->in_desc_->Get(),
        in_diff_blob->mut_dptr<T>()));
  }
}

template<typename T>
void DeconvKernel<DeviceType::kGPU, T>::BiasBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, Blob* bias_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CudaCheck(cudnnConvolutionBackwardBias(device_ctx->cudnn_handle(), OnePtr<T>::value,
                                         this->out_desc_->Get(), out_diff_blob->dptr<T>(),
                                         ZeroPtr<T>::value, this->bias_desc_->Get(),
                                         bias_diff_blob->mut_dptr<T>()));
}

#define INSTANTIATE_DECONV_KERNEL(type_cpp, type_proto) \
  template class DeconvKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_DECONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  //  namespace oneflow
