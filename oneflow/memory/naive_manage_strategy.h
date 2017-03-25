#ifndef _MEMORY_NAIVE_MANAGE_STRATEGY_H_
#define _MEMORY_NAIVE_MANAGE_STRATEGY_H_

#include "memory/base_manage_strategy.h"

namespace oneflow {
template <class DeviceMemory>
class NaiveManageStrategy : public BaseManageStrategy {
 public:
  NaiveManageStrategy() = default;
  ~NaiveManageStrategy() = default;

  void* Alloc(size_t size) override;
  void Free(void* ptr, size_t size) override;
  size_t Size() override;

 private:
  NaiveManageStrategy(const NaiveManageStrategy& other) = delete;
  NaiveManageStrategy& operator= (const NaiveManageStrategy& other) = delete;
};
}  // namespace oneflow
#endif  // _MEMORY_NAIVE_MANAGE_STRATEGY_H_
