#ifndef ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_
#define ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_

#include "oneflow/core/operator/op_infer_cache.h"
#include "oneflow/core/kernel/kernel.pb.h"

namespace oneflow {

namespace user_op {

class KernelInferContext;

class OpKernelInferCacheHelper final {
 public:
  OpKernelInferCacheHelper(const KernelConf& kernel_conf, const JobDesc& job_desc);
  ~OpKernelInferCacheHelper() = default;

  void ForwardShape(std::function<void(KernelInferContext*)> infer_fn, KernelInferContext* ctx);

 private:
  void UpdateCacheKey(KernelInferContext* ctx);

  OpInferCacheKey cache_key_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_
