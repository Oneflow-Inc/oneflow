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
#include <memory>
#include <mutex>
#include <thread>
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace vm {
class ThreadSafeGuard final {
 public:
  ThreadSafeGuard() = default;
  ~ThreadSafeGuard() = default;
  OF_DISALLOW_COPY_AND_MOVE(ThreadSafeGuard);

  std::unique_lock<std::mutex> GetGuard() {
    std::unique_lock<std::mutex> lock(mutex4backend_allocator_);
    return lock;
  }

 private:
  std::mutex mutex4backend_allocator_;
};

class SingleThreadGuard final {
 public:
  SingleThreadGuard() : accessed_thread_id_(std::this_thread::get_id()) {}
  ~SingleThreadGuard() = default;
  OF_DISALLOW_COPY_AND_MOVE(SingleThreadGuard);

  bool GetGuard() {
    std::unique_lock<std::mutex> lock(mutex4accessed_thread_id_);
    CHECK(accessed_thread_id_ == std::this_thread::get_id());
    return true;
  }

 private:
  void CheckUniqueThreadAccess();

  std::thread::id accessed_thread_id_;
  std::mutex mutex4accessed_thread_id_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD_SAFE_ALLOCATOR_H_
