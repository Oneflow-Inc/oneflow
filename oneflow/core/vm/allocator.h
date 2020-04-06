#ifndef ONEFLOW_CORE_VM_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_ALLOCATOR_H_

#include <cstddef>

namespace oneflow {
namespace vm {

class Allocator {
 public:
  virtual ~Allocator() = default;

  virtual void Allocate(char** mem_ptr, std::size_t size) = 0;
  virtual void Deallocate(char* mem_ptr, std::size_t size) = 0;

 protected:
  Allocator() = default;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ALLOCATOR_H_
