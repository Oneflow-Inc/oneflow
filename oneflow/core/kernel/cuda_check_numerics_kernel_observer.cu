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
#include "oneflow/core/kernel/cuda_check_numerics_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__device__ bool IsNotFinite(T x) {
  return !isfinite(x);
}

#if __CUDA_ARCH__ >= 530
template<>
__device__ bool IsNotFinite<half>(half x) {
  return (__hisinf(x) || __hisnan(x));
}
#endif

template<typename T>
__global__ void HasNotFiniteGpuKernel(const int64_t n, const T* x, volatile bool* has_not_finite) {
  if (*has_not_finite) { return; }
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    if (IsNotFinite(x[i])) {
      *has_not_finite = true;
      return;
    }
  }
}

template<typename T>
bool HasNotFinite(ep::Stream* stream, const int64_t elem_cnt, const T* data_ptr,
                  bool* has_not_finite_host, bool* has_not_finite_device) {
  OF_CUDA_CHECK(cudaMemsetAsync(has_not_finite_device, 0, sizeof(bool),
                                stream->As<ep::CudaStream>()->cuda_stream()));
  HasNotFiniteGpuKernel<T>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
         stream->As<ep::CudaStream>()->cuda_stream()>>>(elem_cnt, data_ptr, has_not_finite_device);
  OF_CUDA_CHECK(cudaMemcpyAsync(has_not_finite_host, has_not_finite_device, sizeof(bool),
                                cudaMemcpyDefault, stream->As<ep::CudaStream>()->cuda_stream()));
  OF_CUDA_CHECK(cudaStreamSynchronize(stream->As<ep::CudaStream>()->cuda_stream()));
  return *has_not_finite_host;
}

bool HasNotFiniteGpu(ep::Stream* stream, const Blob* blob, bool* has_not_finite_host,
                     bool* has_not_finite_device) {
  const DataType dtype = blob->data_type();
  const int64_t elem_cnt = blob->shape().elem_cnt();
  if (dtype == kFloat) {
    return HasNotFinite<float>(stream, elem_cnt, blob->dptr<float>(), has_not_finite_host,
                               has_not_finite_device);
  } else if (dtype == kDouble) {
    return HasNotFinite<double>(stream, elem_cnt, blob->dptr<double>(), has_not_finite_host,
                                has_not_finite_device);
  } else if (dtype == kFloat16) {
#if __CUDA_ARCH__ >= 530
    return HasNotFinite<half>(stream, elem_cnt, blob->dptr<half>(), has_not_finite_host,
                              has_not_finite_device);
#else
    LOG(FATAL) << "use half need nvcc arch >= 530";
#endif
  } else {
    return false;
  }
}

}  // namespace

CudaCheckNumericsKernelObserver::CudaCheckNumericsKernelObserver()
    : has_not_finite_host_(nullptr), has_not_finite_device_(nullptr) {
  OF_CUDA_CHECK(cudaGetDevice(&device_id_));
  OF_CUDA_CHECK(cudaMallocHost(&has_not_finite_host_, sizeof(bool)));
  OF_CUDA_CHECK(cudaMalloc(&has_not_finite_device_, sizeof(bool)));
}

CudaCheckNumericsKernelObserver::~CudaCheckNumericsKernelObserver() {
  CudaCurrentDeviceGuard guard(device_id_);
  OF_CUDA_CHECK(cudaFreeHost(has_not_finite_host_));
  OF_CUDA_CHECK(cudaFree(has_not_finite_device_));
}

void CudaCheckNumericsKernelObserver::DidForwardDataContent(KernelContext* ctx,
                                                            const Kernel* kernel) {
  for (const auto& obn : kernel->op_attribute().output_bns()) {
    Blob* blob = ctx->BnInOp2Blob(obn);
    if (blob != nullptr) {
      bool has_not_finite =
          HasNotFiniteGpu(ctx->stream(), blob, has_not_finite_host_, has_not_finite_device_);
      CHECK(!has_not_finite) << kernel->op_conf().name() << " : " << obn << " has nan or inf";
    }
  }
}

}  // namespace oneflow
