#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_INFER_CACHE_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_INFER_CACHE_H_

#include "oneflow/core/operator/op_infer_cache.h"
#include "oneflow/core/common/hash_eq_trait_ptr.h"
#include "oneflow/core/kernel/kernel.pb.h"

namespace oneflow {

namespace user_op {

class KernelInferContext;

class OpKernelInferCache final {
 public:
  using KeyType = OpInferCacheKey;
  using ValueType = std::shared_ptr<const OpInferCacheValue>;
  using HashMap = std::unordered_map<HashEqTraitPtr<const KeyType>, ValueType>;
  using KeyStorage = std::list<std::unique_ptr<KeyType>>;
  static constexpr size_t kReleaseInIndependentThreadThreshold = 4096;

  OpKernelInferCache(const KernelConf& kernel_conf, const JobDesc& job_desc);
  ~OpKernelInferCache() = default;

  bool IsCacheHit() const;
  ValueType GetCacheValue() const;
  void UpdateCacheKey(KernelInferContext* ctx);
  void UpdateCacheValue(KernelInferContext* ctx);
  void Reset();

 private:
  KeyType cache_key_;
  HashMap cached_key2value_;
  KeyStorage key_storage_;
  size_t max_size_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_INFER_CACHE_H_
