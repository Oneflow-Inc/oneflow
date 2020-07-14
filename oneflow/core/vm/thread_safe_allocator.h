#ifndef ONEFLOW_CORE_VM_THREAD_SAFE_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_THREAD_SAFE_ALLOCATOR_H_

#include <cstdint>
#include <mutex>
#include <thread>
#include "oneflow/core/vm/allocator.h"

namespace oneflow {

namespace vm {

class ThreadSafeAllocator final : public Allocator {
 public:
  explicit ThreadSafeAllocator(std::unique_ptr<Allocator>&& backend_allocator)
      : Allocator(), backend_allocator_(std::move(backend_allocator)) {}
  ~ThreadSafeAllocator() override = default;

  void Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;

 private:
  std::unique_ptr<Allocator> backend_allocator_;
  std::mutex mutex4backend_allocator_;
};

class SingleThreadOnlyAllocator final : public Allocator {
 public:
  explicit SingleThreadOnlyAllocator(std::unique_ptr<Allocator>&& backend_allocator)
      : Allocator(),
        backend_allocator_(std::move(backend_allocator)),
        accessed_thread_id_(std::this_thread::get_id()) {}
  ~SingleThreadOnlyAllocator() override = default;

  void Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;

 private:
  void CheckUniqueThreadAccess();

  std::unique_ptr<Allocator> backend_allocator_;
  std::thread::id accessed_thread_id_;
  std::mutex mutex4ccessed_thread_id_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD_SAFE_ALLOCATOR_H_
