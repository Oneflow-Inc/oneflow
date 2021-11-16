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
#ifndef ONEFLOW_CORE_EP_CUDA_CUDA_STREAM_H_
#define ONEFLOW_CORE_EP_CUDA_CUDA_STREAM_H_

#include "oneflow/core/ep/include/stream.h"

#ifdef WITH_CUDA

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

namespace ep {

class CudaStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStream);
  explicit CudaStream(int device_ordinal);
  ~CudaStream() override;

  DeviceType device_type() const override;

  cudaStream_t cuda_stream() const;
  cublasHandle_t cublas_handle() const;
  cudnnHandle_t cudnn_handle() const;

 private:
  cudaStream_t cuda_stream_{};
  cublasHandle_t cublas_handle_{};
  cudnnHandle_t cudnn_handle_{};
  int device_ordinal_;
#if CUBLAS_VERSION >= 11200
  void* workspace_{};
  size_t workspace_size_{};
#endif  // CUBLAS_VERSION >= 11200
};

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_EP_CUDA_CUDA_STREAM_H_
