#ifndef ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_

#ifdef WITH_CUDNN

#include "cudnn.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/operator/op_conf.pb.h"

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

class CudnnTensorDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnTensorDesc);
  CudnnTensorDesc() = delete;
  ~CudnnTensorDesc();

  CudnnTensorDesc(DataType, int64_t n, int64_t c, int64_t h, int64_t w);
  CudnnTensorDesc(DataType, const Shape&);

  const cudnnTensorDescriptor_t& Get() const { return val_; }

 private:
  cudnnTensorDescriptor_t val_;
};

class CudnnFilterDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnFilterDesc);
  CudnnFilterDesc() = delete;
  ~CudnnFilterDesc();

  CudnnFilterDesc(DataType, int64_t k, int64_t c, int64_t h, int64_t w);
  CudnnFilterDesc(DataType, const Shape&);

  const cudnnFilterDescriptor_t& Get() const { return val_; }

 private:
  cudnnFilterDescriptor_t val_;
};

class CudnnConvolutionDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvolutionDesc);
  CudnnConvolutionDesc() = delete;
  ~CudnnConvolutionDesc();

  CudnnConvolutionDesc(DataType, int64_t pad_h, int64_t pad_w, int64_t stride_h,
                       int64_t stride_w, int64_t dilation_h,
                       int64_t dilation_w);
  CudnnConvolutionDesc(DataType, const ConvolutionOpConf& conv_conf);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};

}  // namespace oneflow

#endif  // WITH_CUDNN

#endif  // ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_
