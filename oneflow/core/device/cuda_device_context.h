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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

#ifdef WITH_CUDA

class CudaDeviceCtx : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaDeviceCtx);
  CudaDeviceCtx() = delete;
  ~CudaDeviceCtx() override = default;

  explicit CudaDeviceCtx(CudaStreamHandle* cuda_handler) : cuda_handler_(cuda_handler) {}

  const cudaStream_t& cuda_stream() const override { return *(cuda_handler_->cuda_stream()); }
  const cublasHandle_t& cublas_pmh_handle() const override {
    return *(cuda_handler_->cublas_pmh_handle());
  }
  const cublasHandle_t& cublas_tensor_op_math_handle() const override {
    return *(cuda_handler_->cublas_tensor_op_math_handle());
  }
  const cublasHandle_t& cublas_pmd_handle() const override {
    return *(cuda_handler_->cublas_pmd_handle());
  }
  const cudnnHandle_t& cudnn_handle() const override { return *(cuda_handler_->cudnn_handle()); }

  void SyncDevice() override { OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream())); }

  void AddCallBack(std::function<void()> callback) const override {
    cuda_handler_->AddCallBack(callback);
  }

 protected:
  CudaStreamHandle* cuda_handler_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_
