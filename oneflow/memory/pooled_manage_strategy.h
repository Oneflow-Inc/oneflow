#ifndef _MEMORY_POOLED_MANAGE_STRATEGY_H_
#define _MEMORY_POOLED_MANAGE_STRATEGY_H_

#include <unordered_map>
#include <vector>
#include <mutex>
#include "memory/pooled_manage_strategy.h"


namespace caffe {
template <class DeviceStorage, size_t kMemoryPoolSize>
class PooledManageStrategy final : BaseManageStrategy {
 public:
  PooledManageStrategy() = default;
  ~PooledManageStrategy();

  void* Alloc(size_t size) override;
  void Free(void* ptr, size_t size) override;
  void Size() override;

 private:
  PooledManageStrategy(const PooledManageStrategy& other);
  PooledManageStrategy& operator= (const PooledManageStrategy);
  std::mutex mutex_;
  size_t used_memory_ = 0;
  std::unordered_map <size_t, std::vector<void*>> memory_pool_;
};
}  // namespace caffe
#endif  // _MEMORY_POOLED_MANAGE_STRATEGY_H_
