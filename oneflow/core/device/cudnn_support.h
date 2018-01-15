#ifndef ONEFLOW_CORE_DEVICE_CUDNN_SUPPORT_H_
#define ONEFLOW_CORE_DEVICE_CUDNN_SUPPORT_H_

#ifdef WITH_CUDNN
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob_desc.h"

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
                                 const ConvolutionOpConf&);
  cudnnConvolutionFwdAlgo_t InferFwdAlgo(const cudnnHandle_t*);
  cudnnConvolutionBwdFilterAlgo_t InferBwdFilterAlgo(const cudnnHandle_t*);
  cudnnConvolutionBwdDataAlgo_t InferBwdDataAlgo(const cudnnHandle_t*);
  size_t InferWorkspaceSize(const cudnnHandle_t*);

 private:
  cudnnTensorDescriptor_t in_handle_;
  cudnnTensorDescriptor_t out_handle_;
  cudnnFilterDescriptor_t filter_handle_;
  cudnnConvolutionDescriptor_t conv_handle_;
  cudnnTensorDescriptor_t bias_handle_;
};

}  // namespace oneflow

#endif  // WITH_CUDNN
#endif  // ONEFLOW_CORE_DEVICE_CUDNN_SUPPORT_H_
