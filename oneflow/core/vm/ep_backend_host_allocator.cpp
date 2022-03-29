#include "oneflow/core/vm/ep_backend_host_allocator.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/device.h"

namespace oneflow {

void EpBackendHostAllocator::Allocate(char** mem_ptr, std::size_t size) {
  CHECK_JUST(ep_device_->AllocPinned(allocation_options_, mem_ptr, size));
}

void EpBackendHostAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  CHECK_JUST(ep_device_->FreePinned(allocation_options_, mem_ptr));
}

}
