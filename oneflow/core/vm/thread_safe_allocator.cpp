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
#include "oneflow/core/vm/thread_safe_allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

Maybe<void> ThreadSafeAllocator::Allocate(char** mem_ptr, std::size_t size) {
  std::unique_lock<std::mutex> lock(mutex4backend_allocator_);
  return backend_allocator_->Allocate(mem_ptr, size);
}

void ThreadSafeAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  std::unique_lock<std::mutex> lock(mutex4backend_allocator_);
  backend_allocator_->Deallocate(mem_ptr, size);
}

void ThreadSafeAllocator::Shrink() {
  std::unique_lock<std::mutex> lock(mutex4backend_allocator_);
  auto* cache = dynamic_cast<ShrinkableCache*>(backend_allocator_.get());
  if (cache != nullptr) { cache->Shrink(); }
}

void ThreadSafeAllocator::DeviceReset() {
  std::unique_lock<std::mutex> lock(mutex4backend_allocator_);
  backend_allocator_->DeviceReset();
}

Maybe<void> SingleThreadOnlyAllocator::Allocate(char** mem_ptr, std::size_t size) {
  CheckUniqueThreadAccess();
  return backend_allocator_->Allocate(mem_ptr, size);
}

void SingleThreadOnlyAllocator::Deallocate(char* mem_ptr, std::size_t size) {
  CheckUniqueThreadAccess();
  backend_allocator_->Deallocate(mem_ptr, size);
}

void SingleThreadOnlyAllocator::Shrink() {
  CheckUniqueThreadAccess();
  auto* cache = dynamic_cast<ShrinkableCache*>(backend_allocator_.get());
  if (cache != nullptr) { cache->Shrink(); }
}

void SingleThreadOnlyAllocator::DeviceReset() {
  CheckUniqueThreadAccess();
  backend_allocator_->DeviceReset();
}

void SingleThreadOnlyAllocator::CheckUniqueThreadAccess() {
  std::unique_lock<std::mutex> lock(mutex4accessed_thread_id_);
  CHECK(accessed_thread_id_ == std::this_thread::get_id());
}

}  // namespace vm
}  // namespace oneflow
