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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace {

std::shared_ptr<ep::Device> GetAllocationDevice(const MemoryCase& mem_case) {
  DeviceType device_type = DeviceType::kInvalidDevice;
  size_t device_index = 0;
  if (mem_case.has_host_mem()) {
    device_type = DeviceType::kCPU;
  } else if (mem_case.has_device_cuda_mem()) {
    device_type = DeviceType::kCUDA;
    device_index = mem_case.device_cuda_mem().device_id();
  } else {
    UNIMPLEMENTED();
  }
  auto device = Global<ep::DeviceManagerRegistry>::Get()->GetDevice(device_type, device_index);
  CHECK(device);
  return device;
}

ep::AllocationOptions GetAllocationOptions(const MemoryCase& mem_case) {
  ep::AllocationOptions options{};
  if (mem_case.has_host_mem() && mem_case.host_mem().has_cuda_pinned_mem()) {
    options.SetPinnedDevice(DeviceType::kCUDA, mem_case.host_mem().cuda_pinned_mem().device_id());
  }
  return options;
}

}  // namespace

void* MemoryAllocatorImpl::Allocate(const MemoryCase& mem_case, size_t size) {
  void* ptr = nullptr;
  std::shared_ptr<ep::Device> device = GetAllocationDevice(mem_case);
  ep::AllocationOptions options = GetAllocationOptions(mem_case);
  CHECK_JUST(device->Alloc(options, &ptr, size));
  return ptr;
}

void MemoryAllocatorImpl::Deallocate(void* ptr, const MemoryCase& mem_case) {
  std::shared_ptr<ep::Device> device = GetAllocationDevice(mem_case);
  ep::AllocationOptions options = GetAllocationOptions(mem_case);
  device->Free(options, ptr);
}

void* MemoryAllocatorImpl::AllocateUnPinnedHostMem(size_t size) {
  void* ptr = aligned_alloc(kHostAlignSize, size);
  CHECK_NOTNULL(ptr);
  return ptr;
}

void MemoryAllocatorImpl::DeallocateUnPinnedHostMem(void* ptr) {
  free(ptr);  // NOLINT
}

MemoryAllocator::~MemoryAllocator() {
  for (const std::function<void()>& deleter : deleters_) { deleter(); }
}

char* MemoryAllocator::Allocate(const MemoryCase& mem_case, std::size_t size) {
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

void MemoryAllocator::Deallocate(char* dptr, const MemoryCase& mem_case) {
  MemoryAllocatorImpl::Deallocate(static_cast<void*>(dptr), mem_case);
}

void InitNonPODTypeBlobIfNeed(MemoryAllocator* allocator, Blob* blob_ptr) {
  const BlobDesc& blob_desc = blob_ptr->blob_desc();
  if (blob_desc.data_type() == kOFRecord) {
    int64_t elem_cnt = blob_desc.shape().elem_cnt();
    FOR_RANGE(int64_t, idx, 0, elem_cnt) {
      allocator->PlacementNew(&blob_ptr->mut_dptr<OFRecord>()[idx]);
    }
  }
  if (blob_desc.data_type() == kTensorBuffer) {
    int64_t elem_cnt = blob_desc.shape().elem_cnt();
    FOR_RANGE(int64_t, idx, 0, elem_cnt) {
      allocator->PlacementNew(&blob_ptr->mut_dptr<TensorBuffer>()[idx]);
    }
  }
}

void InitNonPODTypeEagerBlobObjectIfNeed(MemoryAllocator* allocator,
                                         vm::EagerBlobObject* eager_blob_object_ptr) {
  if (eager_blob_object_ptr->data_type() == kOFRecord) {
    int64_t elem_cnt = eager_blob_object_ptr->shape().elem_cnt();
    FOR_RANGE(int64_t, idx, 0, elem_cnt) {
      allocator->PlacementNew(&eager_blob_object_ptr->mut_dptr<OFRecord>()[idx]);
    }
  }
  if (eager_blob_object_ptr->data_type() == kTensorBuffer) {
    int64_t elem_cnt = eager_blob_object_ptr->shape().elem_cnt();
    FOR_RANGE(int64_t, idx, 0, elem_cnt) {
      allocator->PlacementNew(&eager_blob_object_ptr->mut_dptr<TensorBuffer>()[idx]);
    }
  }
}

}  // namespace oneflow
