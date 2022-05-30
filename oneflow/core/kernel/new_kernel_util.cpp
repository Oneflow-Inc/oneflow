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

namespace oneflow {

template<>
void Memcpy<DeviceType::kCPU>(ep::Stream* stream, void* dst, const void* src, size_t sz) {
  if (dst == src) { return; }
  memcpy(dst, src, sz);
}

template<>
void Memset<DeviceType::kCPU>(ep::Stream* stream, void* dst, const char value, size_t sz) {
  memset(dst, value, sz);
}

void WithHostBlobAndStreamSynchronizeEnv(ep::Stream* stream, Blob* blob,
                                         std::function<void(Blob*)> Callback) {
#ifdef WITH_CUDA
  char* host_raw_dptr = nullptr;
  OF_CUDA_CHECK(cudaMallocHost(&host_raw_dptr, blob->AlignedTotalByteSize()));
  Blob host_blob(MemoryCase(), &blob->blob_desc(), host_raw_dptr);
  Callback(&host_blob);
  Memcpy<DeviceType::kCUDA>(stream, blob->mut_dptr(), host_blob.dptr(), blob->ByteSizeOfBlobBody());
  CHECK_JUST(stream->Sync());
  OF_CUDA_CHECK(cudaFreeHost(host_raw_dptr));
#else
  UNIMPLEMENTED();
#endif
}

#ifdef WITH_NPU

template<>
void Memcpy<DeviceType::kNPU>(ep::Stream* stream, void* dst, const void* src, size_t sz) {
  if (dst == src) { return; }
  std::cout<<"Memcpy aclrtMemcpy src"<<src<<" dst "<<dst<<" sz "<<sz<<std::endl;
  // OF_NPU_CHECK(aclrtMemcpyAsync(dst, sz, src, sz, ACL_MEMCPY_HOST_TO_DEVICE,
  //                               stream->As<ep::NpuStream>()->npu_stream()));//dck_caution_here 
  OF_NPU_CHECK(aclrtMemcpy(dst, sz, src, sz, ACL_MEMCPY_HOST_TO_DEVICE));
                                //ep::Stream : /home/HDD/dck/oneflow/python/oneflow/include/oneflow/core/ep/include/stream.h
}

template<>
void Memset<DeviceType::kNPU>(ep::Stream* stream, void* dst, const char value, size_t sz) {
  std::cout<<"Memset aclrtMemsetAsync "<<std::endl;
  OF_NPU_CHECK(aclrtMemsetAsync(dst, sz, value, sz, stream->As<ep::NpuStream>()->npu_stream()));
}

#endif

}  // namespace oneflow
