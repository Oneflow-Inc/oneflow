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
class ThreadSafeLock final {
 public:
  ThreadSafeLock() = default;
  ~ThreadSafeLock() = default;
  OF_DISALLOW_COPY_AND_MOVE(ThreadSafeLock);

  class RAIIGuard final {
   public:
    explicit RAIIGuard(ThreadSafeLock& lock) : guard_(lock.mutex4guard) {}
    ~RAIIGuard() = default;
    OF_DISALLOW_COPY_AND_MOVE(RAIIGuard);

   private:
    std::unique_lock<std::mutex> guard_;
  };

 private:
  std::mutex mutex4guard;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD_SAFE_ALLOCATOR_H_
