#ifdef WITH_CUDNN
#include "oneflow/core/device/cudnn_support.h"

namespace oneflow {

float CudnnDataType<float>::oneval = 1.0f;
float CudnnDataType<float>::zeroval = 0.0f;
const void* CudnnDataType<float>::one =
    static_cast<void*>(&CudnnDataType<float>::oneval);
const void* CudnnDataType<float>::zero =
    static_cast<void*>(&CudnnDataType<float>::zeroval);

double CudnnDataType<double>::oneval = 1.0;
double CudnnDataType<double>::zeroval = 0.0;
const void* CudnnDataType<double>::one =
    static_cast<void*>(&CudnnDataType<double>::oneval);
const void* CudnnDataType<double>::zero =
    static_cast<void*>(&CudnnDataType<double>::zeroval);

CudnnConvolutionDesc::CudnnConvolutionDesc() {
  CudaCheck(cudnnCreateTensorDescriptor(&this->in_handle_));
  CudaCheck(cudnnCreateTensorDescriptor(&this->out_handle_));
  CudaCheck(cudnnCreateFilterDescriptor(&this->filter_handle_));
  CudaCheck(cudnnCreateConvolutionDescriptor(&this->conv_handle_));
  CudaCheck(cudnnCreateTensorDescriptor(&this->bias_handle_));
}

CudnnConvolutionDesc::~CudnnConvolutionDesc() {
  CudaCheck(cudnnDestroyTensorDescriptor(this->bias_handle_));
  CudaCheck(cudnnDestroyConvolutionDescriptor(this->conv_handle_));
  CudaCheck(cudnnDestroyFilterDescriptor(this->filter_handle_));
  CudaCheck(cudnnDestroyTensorDescriptor(this->out_handle_));
  CudaCheck(cudnnDestroyTensorDescriptor(this->in_handle_));
}

void CudnnConvolutionDesc::InitFromBlobDescAndOpConf(
    const BlobDesc* in_blob_desc, const BlobDesc* out_blob_desc,
    const ConvolutionOpConf& conv_conf) const {
  cudnnDataType_t cudnn_data_type;
  switch (in_blob_desc->data_type()) {
    case kFloat: cudnn_data_type = CUDNN_DATA_FLOAT; break;
    case kDouble: cudnn_data_type = CUDNN_DATA_DOUBLE; break;
    default: UNEXPECTED_RUN();
  }

  CudaCheck(cudnnSetTensor4dDescriptor(
      this->in_handle_, CUDNN_TENSOR_NCHW, cudnn_data_type,
      in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
      in_blob_desc->shape().At(2), in_blob_desc->shape().At(3)));
  CudaCheck(cudnnSetTensor4dDescriptor(
      this->out_handle_, CUDNN_TENSOR_NCHW, cudnn_data_type,
      out_blob_desc->shape().At(0), out_blob_desc->shape().At(1),
      out_blob_desc->shape().At(2), out_blob_desc->shape().At(3)));
  CudaCheck(cudnnSetFilter4dDescriptor(
      this->filter_handle_, cudnn_data_type, CUDNN_TENSOR_NCHW,
      out_blob_desc->shape().At(1), in_blob_desc->shape().At(1),
      conv_conf.kernel_h(), conv_conf.kernel_w()));
  CudaCheck(cudnnSetConvolution2dDescriptor(
      this->conv_handle_, conv_conf.pad_h(), conv_conf.pad_w(),
      conv_conf.stride_h(), conv_conf.stride_w(), 1, 1, CUDNN_CROSS_CORRELATION,
      cudnn_data_type));
  if (conv_conf.has_bias_term()) {
    CudaCheck(cudnnSetTensor4dDescriptor(this->bias_handle_, CUDNN_TENSOR_NCHW,
                                         cudnn_data_type, 1,
                                         out_blob_desc->shape().At(1), 1, 1));
  }
}

cudnnConvolutionFwdAlgo_t CudnnConvolutionDesc::InferFwdAlgo(
    const cudnnHandle_t& cudnn_handle) const {
  cudnnConvolutionFwdAlgo_t fwd_algo;

  CudaCheck(cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle, this->in_handle_, this->filter_handle_, this->conv_handle_,
      this->out_handle_, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo));

  return fwd_algo;
}

cudnnConvolutionBwdFilterAlgo_t CudnnConvolutionDesc::InferBwdFilterAlgo(
    const cudnnHandle_t& cudnn_handle) const {
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle, this->in_handle_, this->out_handle_, this->conv_handle_,
      this->filter_handle_, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
      &bwd_filter_algo));

  return bwd_filter_algo;
}

cudnnConvolutionBwdDataAlgo_t CudnnConvolutionDesc::InferBwdDataAlgo(
    const cudnnHandle_t& cudnn_handle) const {
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;

  CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, this->filter_handle_, this->out_handle_, this->conv_handle_,
      this->in_handle_, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
      &bwd_data_algo));

  return bwd_data_algo;
}

size_t CudnnConvolutionDesc::InferWorkspaceSize(
    const cudnnHandle_t& cudnn_handle) const {
  size_t fwd_workspace_sizes = 0;
  size_t bwd_filter_workspace_sizes = 0;
  size_t bwd_data_workspace_sizes = 0;

  // get workspace sizes of algorithm
  CudaCheck(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_handle, this->in_handle_, this->filter_handle_, this->conv_handle_,
      this->out_handle_, this->InferFwdAlgo(cudnn_handle),
      &fwd_workspace_sizes));
  CudaCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_handle, this->in_handle_, this->out_handle_, this->conv_handle_,
      this->filter_handle_, this->InferBwdFilterAlgo(cudnn_handle),
      &bwd_filter_workspace_sizes));
  CudaCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle, this->filter_handle_, this->out_handle_, this->conv_handle_,
      this->in_handle_, this->InferBwdDataAlgo(cudnn_handle),
      &bwd_data_workspace_sizes));

  return std::max({fwd_workspace_sizes, bwd_filter_workspace_sizes,
                   bwd_data_workspace_sizes});
}

}  // namespace oneflow
#endif  // WITH_CUDNN
