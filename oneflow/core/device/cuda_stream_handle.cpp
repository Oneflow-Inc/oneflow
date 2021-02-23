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
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

#ifdef WITH_CUDA

const cudaStream_t* CudaStreamHandle::cuda_stream() {
  if (!cuda_stream_) {
    cuda_stream_.reset(new cudaStream_t);
    OF_CUDA_CHECK(cudaStreamCreate(cuda_stream_.get()));
  }
  return cuda_stream_.get();
}

const cublasHandle_t* CudaStreamHandle::cublas_pmh_handle() {
  if (!cublas_pmh_handle_) {
    cublas_pmh_handle_.reset(new cublasHandle_t);
    OF_CUBLAS_CHECK(cublasCreate(cublas_pmh_handle_.get()));
    OF_CUBLAS_CHECK(cublasSetStream(*cublas_pmh_handle_, *cuda_stream()));
#if CUDA_VERSION >= 11000
    if (Global<ResourceDesc, ForSession>::Get()->enable_tensor_float_32_compute()) {
      OF_CUBLAS_CHECK(cublasSetMathMode(*cublas_pmh_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
    }
#endif
  }
  return cublas_pmh_handle_.get();
}

const cublasHandle_t* CudaStreamHandle::cublas_pmd_handle() {
  if (!cublas_pmd_handle_) {
    cublas_pmd_handle_.reset(new cublasHandle_t);
    OF_CUBLAS_CHECK(cublasCreate(cublas_pmd_handle_.get()));
    OF_CUBLAS_CHECK(cublasSetStream(*cublas_pmd_handle_, *cuda_stream()));
    OF_CUBLAS_CHECK(cublasSetPointerMode(*cublas_pmd_handle_, CUBLAS_POINTER_MODE_DEVICE));
  }
  return cublas_pmd_handle_.get();
}

const cublasHandle_t* CudaStreamHandle::cublas_tensor_op_math_handle() {
  if (!cublas_tensor_op_math_handle_) {
    cublas_tensor_op_math_handle_.reset(new cublasHandle_t);
    OF_CUBLAS_CHECK(cublasCreate(cublas_tensor_op_math_handle_.get()));
    OF_CUBLAS_CHECK(cublasSetStream(*cublas_tensor_op_math_handle_, *cuda_stream()));
#if CUDA_VERSION >= 11000
    OF_CUBLAS_CHECK(cublasSetMathMode(*cublas_tensor_op_math_handle_, CUBLAS_DEFAULT_MATH));
#else
    OF_CUBLAS_CHECK(cublasSetMathMode(*cublas_tensor_op_math_handle_, CUBLAS_TENSOR_OP_MATH));
#endif
  }
  return cublas_tensor_op_math_handle_.get();
}

const cudnnHandle_t* CudaStreamHandle::cudnn_handle() {
  if (!cudnn_handle_) {
    if (IsCuda9OnTuringDevice()) {
      OF_CUDA_CHECK(cudaDeviceSynchronize());
      OF_CUDA_CHECK(cudaGetLastError());
    }
    cudnn_handle_.reset(new cudnnHandle_t);
    OF_CUDNN_CHECK(cudnnCreate(cudnn_handle_.get()));
    if (IsCuda9OnTuringDevice()) {
      OF_CUDA_CHECK(cudaDeviceSynchronize());
      cudaGetLastError();
    }
    OF_CUDNN_CHECK(cudnnSetStream(*cudnn_handle_, *cuda_stream()));
  }
  return cudnn_handle_.get();
}

void CudaStreamHandle::AddCallBack(std::function<void()> callback) {
  CudaCBEvent cb_event;
  cb_event.callback = std::move(callback);
  OF_CUDA_CHECK(cudaEventCreateWithFlags(&(cb_event.event), cudaEventDisableTiming));
  OF_CUDA_CHECK(cudaEventRecord(cb_event.event, *cuda_stream()));
  cb_event_chan_->Send(cb_event);
}

CudaStreamHandle::~CudaStreamHandle() {
  if (cudnn_handle_) { OF_CUDNN_CHECK(cudnnDestroy(*cudnn_handle_)); }
  if (cublas_pmh_handle_) { OF_CUBLAS_CHECK(cublasDestroy(*cublas_pmh_handle_)); }
  if (cublas_pmd_handle_) { OF_CUBLAS_CHECK(cublasDestroy(*cublas_pmd_handle_)); }
  if (cublas_tensor_op_math_handle_) {
    OF_CUBLAS_CHECK(cublasDestroy(*cublas_tensor_op_math_handle_));
  }
  if (cuda_stream_) { OF_CUDA_CHECK(cudaStreamDestroy(*cuda_stream_)); }
}

#endif  // WITH_CUDA

}  // namespace oneflow
