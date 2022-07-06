/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_VM_THREAD_SAFE_ALLOCATOR_H_
#define ONEFLOW_CORE_VM_THREAD_SAFE_ALLOCATOR_H_

#include <cstdint>
#include <mutex>
#include <thread>
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/vm/shrinkable_cache.h"

namespace oneflow {

namespace vm {

class ThreadSafeAllocator final : public Allocator, public ShrinkableCache {
 public:
  explicit ThreadSafeAllocator(std::unique_ptr<Allocator>&& backend_allocator)
      : Allocator(), backend_allocator_(std::move(backend_allocator)) {}
  ~ThreadSafeAllocator() override = default;

  Maybe<void> Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;
  void Shrink() override;
  void DeviceReset() override;

 private:
  std::unique_ptr<Allocator> backend_allocator_;
  std::mutex mutex4backend_allocator_;
};

class SingleThreadOnlyAllocator final : public Allocator, public ShrinkableCache {
 public:
  explicit SingleThreadOnlyAllocator(std::unique_ptr<Allocator>&& backend_allocator)
      : Allocator(),
        backend_allocator_(std::move(backend_allocator)),
        accessed_thread_id_(std::this_thread::get_id()) {}
  ~SingleThreadOnlyAllocator() override = default;

  Maybe<void> Allocate(char** mem_ptr, std::size_t size) override;
  void Deallocate(char* mem_ptr, std::size_t size) override;
  void Shrink() override;
  void DeviceReset() override;

 private:
  void CheckUniqueThreadAccess();

  std::unique_ptr<Allocator> backend_allocator_;
  std::thread::id accessed_thread_id_;
  std::mutex mutex4accessed_thread_id_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD_SAFE_ALLOCATOR_H_
