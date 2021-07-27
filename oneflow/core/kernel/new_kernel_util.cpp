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
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/device/hip_util.hip.h"

namespace oneflow {

template<>
void Memcpy<DeviceType::kCPU>(DeviceCtx* ctx, void* dst, const void* src, size_t sz) {
  if (dst == src) { return; }
  memcpy(dst, src, sz);
}

template<>
void Memset<DeviceType::kCPU>(DeviceCtx* ctx, void* dst, const char value, size_t sz) {
  memset(dst, value, sz);
}

#if defined(WITH_HIP)

template<>
void Memcpy<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const void* src, size_t sz) {
  if (dst == src) { return; }
  OF_HIP_CHECK(hipMemcpyAsync(dst, src, sz, hipMemcpyDefault, ctx->hip_stream()));
}

template<>
void Memset<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const char value, size_t sz) {
  OF_HIP_CHECK(hipMemsetAsync(dst, value, sz, ctx->hip_stream()));
}

#endif

void WithHostBlobAndStreamSynchronizeEnv(DeviceCtx* ctx, Blob* blob,
                                         std::function<void(Blob*)> Callback) {
#ifdef WITH_CUDA
  char* host_raw_dptr = nullptr;
  OF_CUDA_CHECK(cudaMallocHost(&host_raw_dptr, blob->AlignedTotalByteSize()));
  Blob host_blob(MemoryCase(), &blob->blob_desc(), host_raw_dptr);
  Callback(&host_blob);
  Memcpy<DeviceType::kGPU>(ctx, blob->mut_dptr(), host_blob.dptr(), blob->ByteSizeOfBlobBody());
  OF_CUDA_CHECK(cudaStreamSynchronize(ctx->cuda_stream()));
  OF_CUDA_CHECK(cudaFreeHost(host_raw_dptr));
#elif WITH_HIP
  char* host_raw_dptr = nullptr;
  OF_HIP_CHECK(hipHostMalloc(&host_raw_dptr, blob->AlignedTotalByteSize()));
  Blob host_blob(MemoryCase(), &blob->blob_desc(), host_raw_dptr);
  Callback(&host_blob);
  Memcpy<DeviceType::kGPU>(ctx, blob->mut_dptr(), host_blob.dptr(), blob->ByteSizeOfBlobBody());
  OF_HIP_CHECK(hipStreamSynchronize(ctx->hip_stream()));
  OF_HIP_CHECK(hipHostFree(host_raw_dptr));
#else
  UNIMPLEMENTED();
#endif
}



}  // namespace oneflow
