#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/device.h"

namespace oneflow {

void EpBackendAllocator::Allocate(char** mem_ptr, std::size_t size) {
  CHECK_JUST(ep_device_->Alloc(allocation_options_, mem_ptr, size));
}

void EpBackendAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  CHECK_JUST(ep_device_->Free(allocation_options_, mem_ptr));
}

void EpBackendAllocator::DeviceReset() {
#ifdef WITH_CUDA
  if (ep_device_->device_type() == DeviceType::kCUDA) {
    ep_device_->SetAsActiveDevice();
    // NOTE(chengcheng): In some corner case on ubuntu, cuda memory not released even if OOM.
    //   So there need release all cuda memory allocated by this process before core dump.
    LOG(WARNING) << "OOM error is detected, process will exit. And it will start to reset CUDA "
                 << "device for releasing device memory.";
    OF_CUDA_CHECK(cudaDeviceReset());
  }
#endif
}

}
