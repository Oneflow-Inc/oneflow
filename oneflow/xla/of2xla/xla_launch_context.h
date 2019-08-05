#ifndef ONEFLOW_CORE_OF2XLA_XLA_COMPLATION_CONTEXT_H_
#define ONEFLOW_CORE_OF2XLA_XLA_COMPLATION_CONTEXT_H_

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"

#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "oneflow/core/job/resource.pb.h"  // DeviceType
#include "oneflow/core/job/placement.pb.h"  // ParallelContext
#include "oneflow/core/device/device_context.h"
#include "oneflow/xla/of2xla/xla_allocator.h"

namespace oneflow {
namespace mola {

class XlaLaunchResourceMgr {
 public:
  static xla::LocalClient *GetOrCreateLocalClient(
      const se::Platform *platform, int intra_op_num_threads);
  
  static DeviceMemoryPool *GetOrCreateCpuMemoryPool(int device_ordinal);
  static DeviceMemoryPool *GetOrCreateGpuMemoryPool(const void *cuda_stream,
                                                    int device_ordinal);

  static Eigen::ThreadPoolDevice *GetOrCreateEigenHostDevice();
};

class XlaLaunchContext {
 public:
  explicit XlaLaunchContext(const std::string &builder_name,
                            DeviceCtx *device_ctx, 
                            DeviceType device_type,
                            int intra_op_num_threads);
  xla::LocalClient *client() const { return client_; }
  xla::XlaBuilder *builder() const { return builder_.get(); }
  XlaAllocator *allocator() const { return allocator_.get(); }

  Eigen::ThreadPoolDevice *host_device() const { return host_device_; }

  DeviceCtx *device_ctx() const { return device_ctx_; }
  DeviceType device_type() const { return device_type_; }
  int device_ordinal() const { return device_ordinal_; }

  const ParallelContext &parallel_ctx() const { return parallel_ctx_; }

  se::Stream *stream() const { return stream_.get(); }

  void ReserveWorkspace(size_t workspace_bytes);

 private:
  xla::LocalClient *NewLocalClient(const se::Platform *platform,
                                   int num_threads);
  std::shared_ptr<XlaAllocator> NewAllocator(const se::Platform *platform,
                                             int device_ordinal);
  DeviceMemoryPool *NewCpuMemoryPool(int device_ordinal);
  DeviceMemoryPool *NewGpuMemoryPool(const void *cuda_stream,
                                     int device_ordinal);
  Eigen::ThreadPoolDevice* NewEigenHostDevice();

  xla::LocalClient *client_;
  std::shared_ptr<xla::XlaBuilder> builder_;

  xla::StreamPool::Ptr stream_;

  std::shared_ptr<XlaAllocator> allocator_;

  Eigen::ThreadPoolDevice *host_device_;

  DeviceCtx *device_ctx_;
  DeviceType device_type_;
  int device_ordinal_ = -1;

  ParallelContext parallel_ctx_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OF2XLA_XLA_COMPLATION_CONTEXT_H_
