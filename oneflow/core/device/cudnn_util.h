#ifndef ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_

#ifdef WITH_CUDNN

#include "cudnn.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

#define CUDNN_DATA_TYPE_SEQ                       \
  OF_PP_MAKE_TUPLE_SEQ(float, CUDNN_DATA_FLOAT)   \
  OF_PP_MAKE_TUPLE_SEQ(double, CUDNN_DATA_DOUBLE) \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, CUDNN_DATA_INT8)   \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, CUDNN_DATA_INT32)

cudnnDataType_t GetCudnnDataType(DataType);

template<typename T>
struct CudnnDataType;

#define DECLARE_CUDNN_DATA_TYPE(type_cpp, type_cudnn) \
  template<>                                          \
  struct CudnnDataType<type_cpp> {                    \
    static const cudnnDataType_t val = type_cudnn;    \
    static const constexpr type_cpp oneval = 1;       \
    static const constexpr type_cpp zeroval = 0;      \
    static const constexpr void* one = &oneval;       \
    static const constexpr void* zero = &zeroval;     \
  };
OF_PP_FOR_EACH_TUPLE(DECLARE_CUDNN_DATA_TYPE, CUDNN_DATA_TYPE_SEQ);

class CudnnTensorDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnTensorDesc);
  CudnnTensorDesc() = delete;
  ~CudnnTensorDesc();

  CudnnTensorDesc(DataType, int n, int c, int h, int w);
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

  CudnnFilterDesc(DataType, int k, int c, int h, int w);
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

  CudnnConvolutionDesc(DataType, const Conv2dOpConf& conv_conf);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};

}  // namespace oneflow

#endif  // WITH_CUDNN

#endif  // ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_
