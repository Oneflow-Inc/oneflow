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
#ifndef ONEFLOW_CORE_COMMON_THREAD_UNSAFE_OBJ_POOL_H_
#define ONEFLOW_CORE_COMMON_THREAD_UNSAFE_OBJ_POOL_H_

#include <vector>
#include <mutex>
#include <memory>
#include <thread>
#include <glog/logging.h>
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/obj_pool_reuse_strategy.h"

namespace oneflow {
namespace obj_pool {

// object pool for single thread.
template<typename T, ReuseStrategy reuse_strategy = kEnableReconstruct>
class ThreadUnsafeObjPool
    : public std::enable_shared_from_this<ThreadUnsafeObjPool<T, reuse_strategy>> {
 public:
  ThreadUnsafeObjPool() : pool_() { pool_.reserve(kInitPoolCap); }
  ~ThreadUnsafeObjPool() {
    if (reuse_strategy != kEnableReconstruct) {
      for (T* ptr : pool_) { delete ptr; }
    }
  }

  template<typename... Args>
  std::shared_ptr<T> make_shared(Args&&... args) {
    auto* ptr = New(std::forward<Args>(args)...);
    std::weak_ptr<ThreadUnsafeObjPool> pool(this->shared_from_this());
    return std::shared_ptr<T>(ptr, [pool](T* ptr) { TryPut(pool.lock(), ptr); });
  }

 private:
  static constexpr int kInitPoolCap = 1024;

  template<typename... Args>
  T* New(Args&&... args) {
    if (likely(pool_.size())) {
      auto* ptr = Get();
      if (reuse_strategy == kEnableReconstruct) { new (ptr) T(std::forward<Args>(args)...); }
      return ptr;
    }
    return new T(std::forward<Args>(args)...);
  }

  static void TryPut(const std::shared_ptr<ThreadUnsafeObjPool>& pool, T* object) {
    if (likely(static_cast<bool>(pool))) {
      pool->Put(object);
    } else {
      object->~T();
    }
  }

  T* Get() {
    auto* ptr = pool_[pool_.size() - 1];
    pool_.pop_back();
    return ptr;
  }

  void Put(T* obj) {
    pool_.push_back(obj);
    if (reuse_strategy == kEnableReconstruct) { obj->~T(); }
  }

  std::vector<T*> pool_;
};

}  // namespace obj_pool
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_THREAD_UNSAFE_OBJ_POOL_H_
