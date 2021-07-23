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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

template<>
void Memcpy<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const void* src, size_t sz) {
  if (dst == src) { return; }
  OF_CUDA_CHECK(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDefault, ctx->cuda_stream()));
}

template<>
void Memset<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const char value, size_t sz) {
  OF_CUDA_CHECK(cudaMemsetAsync(dst, value, sz, ctx->cuda_stream()));
}

}  // namespace oneflow
