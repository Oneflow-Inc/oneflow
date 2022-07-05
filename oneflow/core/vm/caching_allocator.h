#ifndef ONEFLOW_CORE_VM_CACHING_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_CACHING_ALLOCATOR_H_

#include <cstddef>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/allocator.h"

namespace oneflow {
namespace vm {

class CachingAllocator : public Allocator {
 public:
  virtual ~CachingAllocator() = default;
  virtual void Shrink() = 0;

 protected:
  CachingAllocator() = default;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CACHING_ALLOCATOR_H_
