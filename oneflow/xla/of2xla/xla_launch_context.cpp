#include "tensorflow/compiler/xla/client/client_library.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/memory/device_memory_pool.h"
#include "oneflow/xla/of2xla/xla_launch_context.h"

namespace oneflow {

static ParallelContext LocalParallelContext(int device_ordinal) {
  ParallelContext parallel_ctx;
  parallel_ctx.set_parallel_id(device_ordinal);
  parallel_ctx.set_parallel_num(1);
  parallel_ctx.set_policy(kDataParallel);
  return parallel_ctx;
}

namespace mola {

xla::LocalClient* XlaLaunchResourceMgr::GetOrCreateLocalClient(
    const se::Platform *platform, int intra_op_num_threads) {
  xla::LocalClientOptions client_options;
  client_options.set_platform(const_cast<se::Platform *>(platform));
  client_options.set_intra_op_parallelism_threads(intra_op_num_threads);

  // Get a local client if the client of `client_options` has been created,
  // otherwise create a new local client by `ClientLibrary` and return it
  OF_CHECK_AND_ASSIGN(
      auto client, xla::ClientLibrary::GetOrCreateLocalClient(client_options));
  return client;
}

DeviceBufferAllocator *XlaLaunchResourceMgr::GetOrCreateBufferAllocator(
    const se::Platform *platform, se::Stream *stream, int device_ordinal) {
  static std::unordered_map<uint64_t, DeviceBufferAllocator *> allocators;
  uint64_t stream_id = reinterpret_cast<uint64_t>(stream);

  if (allocators.count(stream_id) == 0) {
    std::shared_ptr<DeviceMemoryPool> mem_pool =
        DeviceMemoryPool::NewMemoryPool(platform, stream, device_ordinal);
    allocators.emplace(stream_id, new DeviceBufferAllocator(mem_pool));
  }
  return allocators[stream_id];
}

Eigen::ThreadPoolDevice* XlaLaunchResourceMgr::GetOrCreateEigenHostDevice() {
  static int default_num_threads = 10;
  static Eigen::ThreadPool threadpool(default_num_threads);
  static Eigen::ThreadPoolDevice host_device(&threadpool,
                                             threadpool.NumThreads());
  return &host_device;
}

xla::LocalClient *XlaLaunchContext::NewLocalClient(
    const se::Platform *platform, int num_threads) {
  return XlaLaunchResourceMgr::GetOrCreateLocalClient(platform, num_threads);
}


XlaAllocator *XlaLaunchContext::NewAllocator(const se::Platform *platform,
                                             int device_ordinal) {
  DeviceBufferAllocator *buffer_allocator =
      XlaLaunchResourceMgr::GetOrCreateBufferAllocator(platform, stream_.get(),
                                                       device_ordinal);
  return new XlaAllocator(platform, buffer_allocator);
}

Eigen::ThreadPoolDevice* XlaLaunchContext::NewEigenHostDevice() {
  return XlaLaunchResourceMgr::GetOrCreateEigenHostDevice();
}

XlaLaunchContext::XlaLaunchContext(const std::string &builder_name,
                                   DeviceCtx *device_ctx,
                                   DeviceType device_type,
                                   int intra_op_num_threads)
    : device_ctx_(device_ctx), device_type_(device_type), device_ordinal_(0) {
  builder_ = std::make_shared<xla::XlaBuilder>(
      absl::StrCat("XlaBuilder_", builder_name));

  se::Platform::Id platform_id = nullptr;
  if (device_type == DeviceType::kCPU) {
    platform_id = se::host::kHostPlatformId;
  } else if (device_type == DeviceType::kGPU) {
    platform_id = se::cuda::kCudaPlatformId;
#ifdef WITH_CUDA
    CudaCheck(cudaGetDevice(&device_ordinal_));
#endif
  }
  CHECK(platform_id) << "Platform Id should not be nullptr. Please check "
                        "your device type.";
  OF_CHECK_AND_ASSIGN(auto platform,
                      se::MultiPlatformManager::PlatformWithId(platform_id));

  client_ = NewLocalClient(platform, intra_op_num_threads);
  OF_CHECK_AND_ASSIGN(stream_,
                      client_->mutable_backend()
                             ->BorrowStream(device_ordinal_));
  allocator_.reset(NewAllocator(platform, device_ordinal_));
  host_device_ = NewEigenHostDevice();
  parallel_ctx_ = LocalParallelContext(device_ordinal_);
}

}  // namespace mola
}  // namespace oneflow
