#ifndef _MEMORY_BASE_MANAGE_STRATEGY_H_
#define _MEMORY_BASE_MANAGE_STRATEGY_H_

#include <cstdint>

namespace caffe {
class BaseManageStrategy {
 public:
  BaseManageStrategy() : alloc_size_(0) {};

  virtual void* Alloc(size_t size) = 0;

  virtual void Free(void* ptr, size_t size) = 0;

  virtual size_t Size() = 0;

  virtual ~BaseManageStrategy() = default;
 protected: 
  size_t alloc_size_;
};
}  // namespace caffe
#endif  // _MEMORY_BASE_MANAGE_STRATEGY_H_