#include "memory/memory_manager.h"
#include <array>
#include "memory/lazy_alloc_array.h"
#include "memory/naive_manage_strategy.h"
#include "memory/base_manage_strategy.h"
#include "device/device_alternate.h"
#include "memory/cpu_device_memory.h"
#include "memory/gpu_device_memory.h"
#include "memory/cuda_pinned_memory.h"

namespace caffe {
class MemoryManagerImpl : public MemoryManager {
 public:
  Handle Alloc(size_t size, Context ctx) override;
  void Free(Handle handle) override;

  MemoryManagerImpl() {}
  virtual ~MemoryManagerImpl() = default;

 private:
  static const size_t kMaxPoolSize = 4096 * 1024 * 1024ul;
  static const size_t kMaxDeviceType = 3;

  template <class DeviceMemory>
  using CurrentManageStrategy =
    NaiveManageStrategy<DeviceMemory>;

  static void SetDevice(Context ctx);

  std::array<LazyAllocArray<BaseManageStrategy>, kMaxDeviceType> memory_mgr_;
};

void MemoryManagerImpl::SetDevice(Context ctx) {
  switch (ctx.dev_type) {
    case DeviceType::kCPU: break;
    case DeviceType::kGPU:
    case DeviceType::kCPUPinned: {
        CUDA_CHECK(cudaSetDevice(ctx.dev_id));
        break;
      }
    default:
      CHECK(0) << "Not implemented device";
  }
}

MemoryManager::Handle MemoryManagerImpl::Alloc(size_t size, Context ctx) {
  Handle hd;
  hd.size = size;
  hd.ctx = ctx;

  auto&& device_mgr = memory_mgr_.at(static_cast<int32_t>(ctx.dev_type));
  BaseManageStrategy* strategy = device_mgr.Get(
    ctx.dev_id, [ctx]() {
      BaseManageStrategy *ptr = nullptr;
      switch (ctx.dev_type) {
        case DeviceType::kCPU: {
          ptr = new CurrentManageStrategy<CPUDeviceMemory>();
          break;
        }
        case DeviceType::kCPUPinned: {
          ptr = new CurrentManageStrategy<CUDAPinnedMemory>();
          break;
        }
        case DeviceType::kGPU: {
          ptr = new CurrentManageStrategy<GPUDeviceMemory>();
          break;
        }
        default: CHECK(0) << "Not implemented device";
      }
      return ptr;
    });
  this->SetDevice(ctx);
  hd.dptr = strategy->Alloc(size);

  return hd;
}

void MemoryManagerImpl::Free(Handle handle) {
  const Context& ctx = handle.ctx;
  auto&& device_mgr = memory_mgr_.at(static_cast<int32_t>(ctx.dev_type));
  BaseManageStrategy* strategy = device_mgr.Get(
    ctx.dev_id, []() {
      CHECK(0) << "Cannot Free space to a device not allocated";
      return nullptr;
    });
  SetDevice(ctx);
  strategy->Free(handle.dptr, handle.size);
}

MemoryManager* MemoryManager::Get() {
  static std::shared_ptr<MemoryManager> inst(new MemoryManagerImpl());
  return inst.get();
}
}  // namespace caffe
