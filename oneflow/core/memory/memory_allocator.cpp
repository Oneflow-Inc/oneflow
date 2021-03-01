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
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/memory/memory_fake_dev_allocator.h"

namespace oneflow {

void* MemoryAllocatorImpl::Allocate(MemoryCase mem_case, size_t size) {
  void* ptr = nullptr;
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
#ifdef WITH_CUDA
      if (Global<ResourceDesc, ForSession>::Get()->enable_numa_aware_cuda_malloc_host()) {
        NumaAwareCudaMallocHost(mem_case.host_mem().cuda_pinned_mem().device_id(), &ptr, size);
      } else {
        OF_CUDA_CHECK(cudaMallocHost(&ptr, size));
      }
#else
      UNIMPLEMENTED();
#endif
    } else {
      ptr = malloc(size);
      CHECK_NOTNULL(ptr);
    }
  } else if (mem_case.has_device_cuda_mem()) {
#ifdef WITH_CUDA
    CudaCurrentDeviceGuard guard(mem_case.device_cuda_mem().device_id());
    OF_CUDA_CHECK(cudaMalloc(&ptr, size));
#else
    UNIMPLEMENTED();
#endif
  } else if (mem_case.has_fake_dev_mem()) {
    ptr = FakeDevMemoryAllocatorImpl::Allocate(mem_case, size);
  } else {
    UNIMPLEMENTED();
  }
  return ptr;
}

void MemoryAllocatorImpl::Deallocate(void* ptr, MemoryCase mem_case) {
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
#ifdef WITH_CUDA
      OF_CUDA_CHECK(cudaFreeHost(ptr));
#else
      UNIMPLEMENTED();
#endif
    } else {
      free(ptr);
    }
  } else if (mem_case.has_device_cuda_mem()) {
#ifdef WITH_CUDA
    CudaCurrentDeviceGuard guard(mem_case.device_cuda_mem().device_id());
    OF_CUDA_CHECK(cudaFree(ptr));
#else
    UNIMPLEMENTED();
#endif
  } else if (mem_case.has_fake_dev_mem()) {
    FakeDevMemoryAllocatorImpl::Deallocate(ptr, mem_case);
  } else {
    UNIMPLEMENTED();
  }
}

void* MemoryAllocatorImpl::AllocateUnPinnedHostMem(size_t size) {
  void* ptr = malloc(size);
  CHECK_NOTNULL(ptr);
  return ptr;
}

void MemoryAllocatorImpl::DeallocateUnPinnedHostMem(void* ptr) { free(ptr); }

MemoryAllocator::~MemoryAllocator() {
  for (std::function<void()> deleter : deleters_) { deleter(); }
}

char* MemoryAllocator::Allocate(MemoryCase mem_case, std::size_t size) {
  const int memset_val = 0;
  char* dptr = static_cast<char*>(MemoryAllocatorImpl::Allocate(mem_case, size));
  if (mem_case.has_host_mem()) {
    memset(dptr, memset_val, size);
  } else if (mem_case.has_device_cuda_mem()) {
#ifdef WITH_CUDA
    CudaCurrentDeviceGuard guard(mem_case.device_cuda_mem().device_id());
    OF_CUDA_CHECK(cudaMemset(dptr, memset_val, size));
#else
    UNIMPLEMENTED();
#endif
  } else {
    UNIMPLEMENTED();
  }
  deleters_.push_front(std::bind(&MemoryAllocator::Deallocate, this, dptr, mem_case));
  return dptr;
}

void MemoryAllocator::Deallocate(char* dptr, MemoryCase mem_case) {
  MemoryAllocatorImpl::Deallocate(static_cast<void*>(dptr), mem_case);
}

void InitNonPODTypeBlobIfNeed(MemoryAllocator* allocator, Blob* blob_ptr) {
  const RtBlobDesc& blob_desc = blob_ptr->blob_desc();
  if (blob_desc.data_type() == kOFRecord) {
    int64_t elem_cnt = blob_desc.body_shape().elem_cnt();
    FOR_RANGE(int64_t, idx, 0, elem_cnt) {
      allocator->PlacementNew(&blob_ptr->mut_dptr<OFRecord>()[idx]);
    }
  }
  if (blob_desc.data_type() == kTensorBuffer) {
    int64_t elem_cnt = blob_desc.body_shape().elem_cnt();
    FOR_RANGE(int64_t, idx, 0, elem_cnt) {
      allocator->PlacementNew(&blob_ptr->mut_dptr<TensorBuffer>()[idx]);
    }
  }
}

}  // namespace oneflow
