/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"

#ifdef WITH_CUDA

#include "cudnn.h"

namespace oneflow {

#define CUDNN_DATA_TYPE_SEQ                       \
  OF_PP_MAKE_TUPLE_SEQ(float, CUDNN_DATA_FLOAT)   \
  OF_PP_MAKE_TUPLE_SEQ(float16, CUDNN_DATA_HALF)  \
  OF_PP_MAKE_TUPLE_SEQ(double, CUDNN_DATA_DOUBLE) \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, CUDNN_DATA_INT8)   \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, CUDNN_DATA_INT32)

cudnnDataType_t GetCudnnDataType(DataType);

template<typename T>
struct CudnnDataType;

#define SPECIALIZE_CUDNN_DATA_TYPE(type_cpp, type_cudnn) \
  template<>                                             \
  struct CudnnDataType<type_cpp> : std::integral_constant<cudnnDataType_t, type_cudnn> {};
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_CUDNN_DATA_TYPE, CUDNN_DATA_TYPE_SEQ);
#undef SPECIALIZE_CUDNN_DATA_TYPE

class CudnnTensorDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnTensorDesc);
  CudnnTensorDesc();
  ~CudnnTensorDesc();

  CudnnTensorDesc(cudnnTensorFormat_t, DataType, int n, int c, int h, int w);
  CudnnTensorDesc(DataType data_type, int dims, const int* dim, const int* stride);
  CudnnTensorDesc(DataType data_type, const ShapeView& shape, const std::string& data_format);

  const cudnnTensorDescriptor_t& Get() const { return val_; }

 private:
  cudnnTensorDescriptor_t val_;
};

class CudnnFilterDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnFilterDesc);
  CudnnFilterDesc() = delete;
  ~CudnnFilterDesc();

  CudnnFilterDesc(DataType data_type, const ShapeView& shape, const std::string& data_format);

  const cudnnFilterDescriptor_t& Get() const { return val_; }

 private:
  cudnnFilterDescriptor_t val_;
};

class CudnnActivationDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnActivationDesc);
  CudnnActivationDesc() = delete;
  ~CudnnActivationDesc();

  CudnnActivationDesc(cudnnActivationMode_t mode, cudnnNanPropagation_t relu_nan_opt, double coef);

  const cudnnActivationDescriptor_t& Get() const { return val_; }

 private:
  cudnnActivationDescriptor_t val_;
};

size_t GetCudnnDataTypeByteSize(cudnnDataType_t data_type);

// SP for scaling parameter
template<typename T>
const void* CudnnSPOnePtr();

template<typename T>
const void* CudnnSPZeroPtr();

const void* CudnnSPOnePtr(const DataType dtype);

const void* CudnnSPZeroPtr(const DataType dtype);

class CudnnHandlePool {
 public:
  CudnnHandlePool() = default;
  ~CudnnHandlePool();
  cudnnHandle_t Get();
  void Put(cudnnHandle_t handle);

 private:
  std::mutex mutex_;
  HashMap<int64_t, std::vector<cudnnHandle_t>> handle_list_map_;
};

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDNN_UTIL_H_
