#include <mutex>

#include "oneflow/xrt/xla/xla_resource_manager.h"

#include "oneflow/xrt/xla/xla_macro.h"
#include "tensorflow/compiler/xla/client/client_library.h"

namespace oneflow {
namespace xrt {
namespace mola {

namespace resource_mgr {

se::Platform::Id GetPlatformId(const XrtDevice &device) {
  se::Platform::Id platform_id = nullptr;
  if (device == XrtDevice::CPU_X86) {
    platform_id = se::host::kHostPlatformId;
  } else if (device == XrtDevice::GPU_CUDA) {
    platform_id = se::cuda::kCudaPlatformId;
  } else {
    LOG(FATAL) << "Only CPU_X86 or GPU_CUDA is supported by XLA.";
  }
  CHECK(platform_id) << "Platform Id should not be nullptr. Current device is " << device
                     << ", please check it.";
  return std::move(platform_id);
}

const se::Platform *GetPlatform(const XrtDevice &device) {
  se::Platform::Id platform_id = GetPlatformId(device);
  MOLA_CHECK_AND_ASSIGN(auto platform, se::MultiPlatformManager::PlatformWithId(platform_id));
  return std::move(platform);
}

Eigen::ThreadPoolDevice *GetOrCreateEigenHostDevice() {
  static int default_num_threads = 10;
  static Eigen::ThreadPool threadpool(default_num_threads);
  static Eigen::ThreadPoolDevice host_device(&threadpool, threadpool.NumThreads());
  return &host_device;
}

DeviceBufferAllocator *GetOrCreateBufferAllocator(const XrtDevice &device,
                                                  const StreamId &stream_id, se::Stream *stream,
                                                  int device_ordinal) {
  static std::unordered_map<StreamId, DeviceBufferAllocator *> buffer_allocators;
  static std::mutex mutex;
  const se::Platform *platform = GetPlatform(device);
  if (buffer_allocators.count(stream_id) == 0) {
    std::lock_guard<std::mutex> lock(mutex);
    while (buffer_allocators.count(stream_id) == 0) {
      std::shared_ptr<DeviceMemoryPool> mem_pool =
          DeviceMemoryPool::NewMemoryPool(platform, stream, device_ordinal);
      buffer_allocators.emplace(stream_id, new DeviceBufferAllocator(mem_pool));
    }
  }
  return buffer_allocators.at(stream_id);
}

xla::LocalClient *GetOrCreateLocalClient(const XrtDevice &device) {
  const se::Platform *platform = GetPlatform(device);
  xla::LocalClientOptions client_options;
  client_options.set_platform(const_cast<se::Platform *>(platform));
  client_options.set_intra_op_parallelism_threads(1);

  // Get a local client if the client of this `client_options` has been created,
  // otherwise create a new local client by `ClientLibrary` and return it.
  MOLA_CHECK_AND_ASSIGN(auto client, xla::ClientLibrary::GetOrCreateLocalClient(client_options));
  return std::move(client);
}

}  // namespace resource_mgr

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
