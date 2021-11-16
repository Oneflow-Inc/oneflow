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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace ep {

namespace {

constexpr size_t kDefaultWorkspaceSize = 4 * 1024 * 1024;  // 4M

}

CudaStream::CudaStream(int device_ordinal) : device_ordinal_(device_ordinal) {
  CudaCurrentDeviceGuard guard(device_ordinal_);
  // cuda_stream
  OF_CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
  // cublas_handle
  OF_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  OF_CUBLAS_CHECK(cublasSetStream(cublas_handle_, cuda_stream_));
#if CUBLAS_VERSION >= 11000
  if (Global<ResourceDesc, ForSession>::Get()->enable_tensor_float_32_compute()) {
    OF_CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
  }
#endif  // CUBLAS_VERSION >= 11000
#if CUBLAS_VERSION >= 11200
  workspace_size_ = kDefaultWorkspaceSize;
  OF_CUDA_CHECK(cudaMalloc(&workspace_, workspace_size_));
  OF_CUBLAS_CHECK(cublasSetWorkspace(cublas_handle_, workspace_, workspace_size_));
#endif  // CUBLAS_VERSION >= 11200
  // cudnn_handle
  if (IsCuda9OnTuringDevice()) {
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    OF_CUDA_CHECK(cudaGetLastError());
  }
  OF_CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
  if (IsCuda9OnTuringDevice()) {
    OF_CUDA_CHECK(cudaDeviceSynchronize());
    cudaGetLastError();
  }
  OF_CUDNN_CHECK(cudnnSetStream(cudnn_handle_, cuda_stream_));
}

CudaStream::~CudaStream() {
  CudaCurrentDeviceGuard guard(device_ordinal_);
  OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  OF_CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
  OF_CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  OF_CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
#if CUBLAS_VERSION >= 11200
  OF_CUDA_CHECK(cudaFree(workspace_));
#endif  // CUBLAS_VERSION >= 11200
}

DeviceType CudaStream::device_type() const { return DeviceType::kGPU; }

cudaStream_t CudaStream::cuda_stream() const { return cuda_stream_; }

cublasHandle_t CudaStream::cublas_handle() const { return cublas_handle_; }

cudnnHandle_t CudaStream::cudnn_handle() const { return cudnn_handle_; }

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
