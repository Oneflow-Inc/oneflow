#ifndef ONEFLOW_CORE_OF2XLA_XLA_COMPLATION_CONTEXT_H_
#define ONEFLOW_CORE_OF2XLA_XLA_COMPLATION_CONTEXT_H_

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "oneflow/core/job/resource.pb.h"  // DeviceType
#include "oneflow/core/job/placement.pb.h"  // ParallelContext
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/compiler/of2xla/xla_allocator.h"

namespace oneflow {
namespace mola {

class CompilationResourceMgr {
 public:
  static xla::LocalClient *GetOrCreateLocalClient(
      const se::Platform *platform, int intra_op_num_threads);
  
  static mola::XlaAllocator *GetOrCreateAllocator(
      const se::Platform *platform);

  static Eigen::ThreadPoolDevice *GetOrCreateEigenHostDevice();
};

class CompilationContext {
 public:
  explicit CompilationContext(const std::string &builder_name,
                              DeviceCtx *device_ctx, 
                              DeviceType device_type,
                              int intra_op_num_threads);

  xla::LocalClient *client() const { return client_; }
  xla::XlaBuilder *builder() const { return builder_.get(); }
  mola::XlaAllocator *allocator() const { return allocator_; }

  Eigen::ThreadPoolDevice *host_device() const { return host_device_; }

  DeviceCtx *device_ctx() const { return device_ctx_; }

  const ParallelContext &parallel_ctx() const { return parallel_ctx_; }

 private:
  xla::LocalClient *client_;
  std::shared_ptr<xla::XlaBuilder> builder_;

  mola::XlaAllocator *allocator_;

  Eigen::ThreadPoolDevice *host_device_;

  DeviceCtx *device_ctx_;
  ParallelContext parallel_ctx_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OF2XLA_XLA_COMPLATION_CONTEXT_H_
