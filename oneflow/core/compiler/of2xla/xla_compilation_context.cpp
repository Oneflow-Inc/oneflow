#include "tensorflow/compiler/xla/client/client_library.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_compilation_context.h"

namespace oneflow {

static ParallelContext LocalParallelContext(DeviceType device_type) {
  int device_id = 0;
  if (device_type == DeviceType::kGPU) {
#ifdef WITH_CUDA
    CudaCheck(cudaGetDevice(&device_id));
#endif
  }
  ParallelContext parallel_ctx;
  parallel_ctx.set_parallel_id(device_id);
  parallel_ctx.set_parallel_num(1);
  parallel_ctx.set_policy(kDataParallel);
  return parallel_ctx;
}

namespace mola {

xla::LocalClient* CompilationResourceMgr::GetOrCreateLocalClient(
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

mola::XlaAllocator* CompilationResourceMgr::GetOrCreateAllocator(
    const se::Platform *platform) {
  static std::unordered_map<se::Platform::Id,
                            std::shared_ptr<mola::XlaAllocator>> allocators;
  se::Platform::Id platform_id = platform->id();
  if (allocators.count(platform_id) == 0) {
    allocators.emplace(platform_id,
                       std::make_shared<mola::XlaAllocator>(platform));
  }
  return allocators[platform_id].get();
}

Eigen::ThreadPoolDevice* CompilationResourceMgr::GetOrCreateEigenHostDevice() {
  static int default_num_threads = 1;
  static Eigen::ThreadPool threadpool(default_num_threads);
  static Eigen::ThreadPoolDevice host_device(&threadpool,
                                             threadpool.NumThreads());
  return &host_device;
}

CompilationContext::CompilationContext(const std::string &builder_name,
                                       DeviceType device_type,
                                       int intra_op_num_threads) {
  typedef CompilationResourceMgr ResourceMgr;
  se::Platform::Id platform_id = nullptr;

  if (device_type == DeviceType::kCPU) {
    platform_id = se::host::kHostPlatformId;
  } else if (device_type == DeviceType::kGPU) {
    platform_id = se::cuda::kCudaPlatformId;
  }
  CHECK(platform_id) << "Platform Id should not be nullptr. Please check "
                        "your device type.";
  OF_CHECK_AND_ASSIGN(auto platform,
                      se::MultiPlatformManager::PlatformWithId(platform_id));

  client_ = ResourceMgr::GetOrCreateLocalClient(platform, intra_op_num_threads);
  builder_ = std::make_shared<xla::XlaBuilder>(absl::StrCat("XlaBuilder_",
                                               builder_name));
  allocator_ = ResourceMgr::GetOrCreateAllocator(platform);
  host_device_ = ResourceMgr::GetOrCreateEigenHostDevice();
  parallel_ctx_ = LocalParallelContext(device_type);
}

}  // namespace mola
}  // namespace oneflow
