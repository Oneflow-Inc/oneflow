#ifndef ONEFLOW_CORE_DEVICE_CUDNN_SUPPORT_H_
#define ONEFLOW_CORE_DEVICE_CUDNN_SUPPORT_H_

#ifdef WITH_CUDNN
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

template<typename T>
struct CudnnDataType;

template<>
struct CudnnDataType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval;
  static float zeroval;
  static const void* one;
  static const void* zero;
};

template<>
struct CudnnDataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval;
  static double zeroval;
  static const void* one;
  static const void* zero;
};

class CudnnConvolutionDesc {
 public:
  CudnnConvolutionDesc();
  ~CudnnConvolutionDesc();

  void InitFromBlobDescAndOpConf(const BlobDesc*, const BlobDesc*,
                                 const ConvolutionOpConf&) const;
  cudnnConvolutionFwdAlgo_t InferFwdAlgo(const cudnnHandle_t&) const;
  cudnnConvolutionBwdFilterAlgo_t InferBwdFilterAlgo(
      const cudnnHandle_t&) const;
  cudnnConvolutionBwdDataAlgo_t InferBwdDataAlgo(const cudnnHandle_t&) const;
  size_t InferWorkspaceSize(const cudnnHandle_t&) const;

  template<typename T>
  void Forward(const cudnnHandle_t&, const Blob*, const Blob*, const Blob*,
               Blob*, Blob*, cudnnConvolutionFwdAlgo_t, bool) const;
  template<typename T>
  void BackwardData(const cudnnHandle_t&, const Blob*, const Blob*, Blob*,
                    Blob*, cudnnConvolutionBwdDataAlgo_t) const;
  template<typename T>
  void BackwardFilter(const cudnnHandle_t&, const Blob*, const Blob*, Blob*,
                      Blob*, cudnnConvolutionBwdFilterAlgo_t) const;
  template<typename T>
  void BackwardBias(const cudnnHandle_t&, const Blob*, Blob*) const;

 private:
  cudnnTensorDescriptor_t in_handle_;
  cudnnTensorDescriptor_t out_handle_;
  cudnnFilterDescriptor_t filter_handle_;
  cudnnConvolutionDescriptor_t conv_handle_;
  cudnnTensorDescriptor_t bias_handle_;
};

template<typename T>
void CudnnConvolutionDesc::Forward(const cudnnHandle_t& cudnn_handle,
                                   const Blob* in_blob, const Blob* weight_blob,
                                   const Blob* bias_blob, Blob* out_blob,
                                   Blob* cudnn_workspace,
                                   cudnnConvolutionFwdAlgo_t cudnn_fwd_algo,
                                   bool has_bias_term) const {
  CudaCheck(cudnnConvolutionForward(
      cudnn_handle, CudnnDataType<T>::one, this->in_handle_, in_blob->dptr<T>(),
      this->filter_handle_, weight_blob->dptr<T>(), this->conv_handle_,
      cudnn_fwd_algo, cudnn_workspace->mut_dptr<T>(),
      cudnn_workspace->shape().At(0), CudnnDataType<T>::zero, this->out_handle_,
      out_blob->mut_dptr<T>()));

  if (has_bias_term) {
    CudaCheck(cudnnAddTensor(cudnn_handle, CudnnDataType<T>::one,
                             this->bias_handle_, bias_blob->dptr<T>(),
                             CudnnDataType<T>::one, this->out_handle_,
                             out_blob->mut_dptr<T>()));
  }
}

template<typename T>
void CudnnConvolutionDesc::BackwardData(
    const cudnnHandle_t& cudnn_handle, const Blob* weight_blob,
    const Blob* out_diff_blob, Blob* in_diff_blob, Blob* cudnn_workspace,
    cudnnConvolutionBwdDataAlgo_t cudnn_bwd_data_algo) const {
  CudaCheck(cudnnConvolutionBackwardData(
      cudnn_handle, CudnnDataType<T>::one, this->filter_handle_,
      weight_blob->dptr<T>(), this->out_handle_, out_diff_blob->dptr<T>(),
      this->conv_handle_, cudnn_bwd_data_algo, cudnn_workspace->mut_dptr<T>(),
      cudnn_workspace->shape().At(0), CudnnDataType<T>::zero, this->in_handle_,
      in_diff_blob->mut_dptr<T>()));
}

template<typename T>
void CudnnConvolutionDesc::BackwardFilter(
    const cudnnHandle_t& cudnn_handle, const Blob* in_blob,
    const Blob* out_diff_blob, Blob* weight_diff_blob, Blob* cudnn_workspace,
    cudnnConvolutionBwdFilterAlgo_t cudnn_bwd_filter_algo) const {
  CudaCheck(cudnnConvolutionBackwardFilter(
      cudnn_handle, CudnnDataType<T>::one, this->in_handle_, in_blob->dptr<T>(),
      this->out_handle_, out_diff_blob->dptr<T>(), this->conv_handle_,
      cudnn_bwd_filter_algo, cudnn_workspace->mut_dptr<T>(),
      cudnn_workspace->shape().At(0), CudnnDataType<T>::one,
      this->filter_handle_, weight_diff_blob->mut_dptr<T>()));
}

template<typename T>
void CudnnConvolutionDesc::BackwardBias(const cudnnHandle_t& cudnn_handle,
                                        const Blob* out_diff_blob,
                                        Blob* bias_diff_blob) const {
  CudaCheck(cudnnConvolutionBackwardBias(
      cudnn_handle, CudnnDataType<T>::one, this->out_handle_,
      out_diff_blob->dptr<T>(), CudnnDataType<T>::one, this->bias_handle_,
      bias_diff_blob->mut_dptr<T>()));
}

}  // namespace oneflow

#endif  // WITH_CUDNN
#endif  // ONEFLOW_CORE_DEVICE_CUDNN_SUPPORT_H_
