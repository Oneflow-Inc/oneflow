#include "oneflow/core/vm/thread_safe_allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void ThreadSafeAllocator::Allocate(char** mem_ptr, std::size_t size) {
  std::unique_lock<std::mutex> lock(mutex4backend_allocator_);
  backend_allocator_->Allocate(mem_ptr, size);
}

void ThreadSafeAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  std::unique_lock<std::mutex> lock(mutex4backend_allocator_);
  backend_allocator_->Deallocate(mem_ptr, size);
}

void SingleThreadOnlyAllocator::Allocate(char** mem_ptr, std::size_t size) {
  CheckUniqueThreadAccess();
  backend_allocator_->Allocate(mem_ptr, size);
}

void SingleThreadOnlyAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  CheckUniqueThreadAccess();
  backend_allocator_->Deallocate(mem_ptr, size);
}

void SingleThreadOnlyAllocator::CheckUniqueThreadAccess() {
  std::unique_lock<std::mutex> lock(mutex4ccessed_thread_id_);
  CHECK(accessed_thread_id_ == std::this_thread::get_id());
}

}  // namespace vm
}  // namespace oneflow
