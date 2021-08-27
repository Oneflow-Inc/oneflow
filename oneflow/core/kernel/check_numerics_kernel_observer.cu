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
#include "oneflow/core/kernel/check_numerics_kernel_observer.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__device__ bool IsNotFinite(T x) {
  return !isfinite(x);
}

template<>
__device__ bool IsNotFinite<half>(half x) {
  return (__hisinf(x) || __hisnan(x));
}

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

class HasNotFiniteContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HasNotFiniteContext);
  HasNotFiniteContext() = default;
  ~HasNotFiniteContext() {
    if (has_not_finite_host_ != nullptr) { OF_CUDA_CHECK(cudaFreeHost(has_not_finite_host_)); }
    if (has_not_finite_device_ != nullptr) { OF_CUDA_CHECK(cudaFree(has_not_finite_device_)); }
  }

  bool* has_not_finite_host() {
    if (has_not_finite_host_ == nullptr) {
      OF_CUDA_CHECK(cudaMallocHost(&has_not_finite_host_, sizeof(bool)));
    }
    return has_not_finite_host_;
  }
  bool* has_not_finite_device() {
    if (has_not_finite_device_ == nullptr) {
      OF_CUDA_CHECK(cudaMalloc(&has_not_finite_device_, sizeof(bool)));
    }
    return has_not_finite_device_;
  }

 private:
  bool* has_not_finite_host_;
  bool* has_not_finite_device_;
};

template<typename T>
bool HasNotFinite(DeviceCtx* device_ctx, const int64_t elem_cnt, const T* data_ptr) {
  thread_local static HasNotFiniteContext ctx;
  OF_CUDA_CHECK(
      cudaMemsetAsync(ctx.has_not_finite_device(), 0, sizeof(bool), device_ctx->cuda_stream()));
  HasNotFiniteGpuKernel<T>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, device_ctx->cuda_stream()>>>(
          elem_cnt, data_ptr, ctx.has_not_finite_device());
  OF_CUDA_CHECK(cudaMemcpyAsync(ctx.has_not_finite_host(), ctx.has_not_finite_device(),
                                sizeof(bool), cudaMemcpyDefault, device_ctx->cuda_stream()));
  OF_CUDA_CHECK(cudaStreamSynchronize(device_ctx->cuda_stream()));
  return *ctx.has_not_finite_host();
}

}  // namespace

bool HasNotFiniteGpu(DeviceCtx* device_ctx, const Blob* blob) {
  const DataType dtype = blob->data_type();
  const int64_t elem_cnt = blob->shape().elem_cnt();
  if (dtype == kFloat) {
    return HasNotFinite<float>(device_ctx, elem_cnt, blob->dptr<float>());
  } else if (dtype == kDouble) {
    return HasNotFinite<double>(device_ctx, elem_cnt, blob->dptr<double>());
  } else if (dtype == kFloat16) {
    return HasNotFinite<half>(device_ctx, elem_cnt, blob->dptr<half>());
  } else {
    return false;
  }
}

}  // namespace oneflow
