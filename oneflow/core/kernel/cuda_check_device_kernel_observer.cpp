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
#include "oneflow/core/kernel/cuda_check_device_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {

CudaCheckDeviceKernelObserver::CudaCheckDeviceKernelObserver(int device_id, cublasMath_t cublas_math_mode, cublasPointerMode_t cublas_pointer_mode, cudaStream_t cublas_stream)
    : device_id_(device_id), cublas_math_mode_(cublas_math_mode), cublas_pointer_mode_(cublas_pointer_mode), cublas_stream_(cublas_stream) {}

void CudaCheckDeviceKernelObserver::DidInit(KernelContext* kernel_ctx, const Kernel* kernel) {
#ifdef WITH_CUDA
  int device_id;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  CHECK_EQ(device_id_, device_id) << kernel->op_conf().name() << " has set cuda device";
  cublasHandle_t cublas_handle =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(kernel_ctx->stream_ctx()))->cublas_handle();
  cublasMath_t cublas_math_mode;
  OF_CUBLAS_CHECK(cublasGetMathMode(cublas_handle, &cublas_math_mode));
  CHECK(cublas_math_mode_ == cublas_math_mode) << kernel->op_conf().name() << " has set cublas math_mode";
  cublasPointerMode_t cublas_pointer_mode{};
  OF_CUBLAS_CHECK(cublasGetPointerMode(cublas_handle, &cublas_pointer_mode));
  CHECK(cublas_pointer_mode_ == cublas_pointer_mode) << kernel->op_conf().name() << " has set cublas pointer_mode";
  cudaStream_t cublas_stream{};
  OF_CUBLAS_CHECK(cublasGetStream(cublas_handle, &cublas_stream));
  CHECK(cublas_stream_ == cublas_stream) << kernel->op_conf().name() << " has set cublas stream";
#endif  // WITH_CUDA
}

void CudaCheckDeviceKernelObserver::DidForward(KernelContext* kernel_ctx, const Kernel* kernel) {
#ifdef WITH_CUDA
  int device_id;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  CHECK_EQ(device_id_, device_id) << kernel->op_conf().name() << " has set cuda device";
  cublasHandle_t cublas_handle =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(kernel_ctx->stream_ctx()))->cublas_handle();
  cublasMath_t cublas_math_mode;
  OF_CUBLAS_CHECK(cublasGetMathMode(cublas_handle, &cublas_math_mode));
  CHECK(cublas_math_mode_ == cublas_math_mode) << kernel->op_conf().name() << " has set cublas math_mode";
  cublasPointerMode_t cublas_pointer_mode{};
  OF_CUBLAS_CHECK(cublasGetPointerMode(cublas_handle, &cublas_pointer_mode));
  CHECK(cublas_pointer_mode_ == cublas_pointer_mode) << kernel->op_conf().name() << " has set cublas pointer_mode";
  cudaStream_t cublas_stream{};
  OF_CUBLAS_CHECK(cublasGetStream(cublas_handle, &cublas_stream));
  CHECK(cublas_stream_ == cublas_stream) << kernel->op_conf().name() << " has set cublas stream";
#endif  // WITH_CUDA
}

}  // namespace oneflow
