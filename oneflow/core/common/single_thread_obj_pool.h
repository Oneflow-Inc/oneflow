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
#ifndef ONEFLOW_CORE_COMMON_SINGLE_THREAD_OBJ_POOL_H_
#define ONEFLOW_CORE_COMMON_SINGLE_THREAD_OBJ_POOL_H_

#include <vector>
#include <mutex>
#include <memory>
#include <thread>
#include <glog/logging.h>
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {
namespace obj_pool {

enum ReuseStrategy {
  kEnableReconstruct,
  kDisableReconstruct,
};

// object pool for single thread.
template<typename T, ReuseStrategy reuse_strategy = kEnableReconstruct>
class SingleThreadObjPool
    : public std::enable_shared_from_this<SingleThreadObjPool<T, reuse_strategy>> {
 public:
  SingleThreadObjPool() : pool_(), invalid_thread_id_(), owner_thread_id_(invalid_thread_id_) {
    pool_.reserve(kInitPoolCap);
  }
  ~SingleThreadObjPool() {
    if (reuse_strategy != kEnableReconstruct) {
      for (T* ptr : pool_) { delete ptr; }
    }
  }

  template<typename... Args>
  std::shared_ptr<T> make_shared(Args&&... args) {
    auto* ptr = New(std::forward<Args>(args)...);
    std::weak_ptr<SingleThreadObjPool> pool(this->shared_from_this());
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

  static void TryPut(const std::shared_ptr<SingleThreadObjPool>& pool, T* object) {
    if (likely(static_cast<bool>(pool))) {
      pool->Put(object);
    } else {
      object->~T();
    }
  }

  T* Get() {
    CheckOrSetSingleThreadFlag();
    auto* ptr = pool_[pool_.size() - 1];
    pool_.pop_back();
    return ptr;
  }

  void Put(T* obj) {
    CheckOrSetSingleThreadFlag();
    pool_.push_back(obj);
    if (reuse_strategy == kEnableReconstruct) { obj->~T(); }
  }

  // Try to detect being wrongly used by multi threads, because SingleThreadObjPool does not
  // guarantee thread safety. This function also is not thread safe, but it's not a big problem. In
  // the most cases, bugs will be successfully detected even thread unsafe behaviors happen.
  void CheckOrSetSingleThreadFlag() {
    if (unlikely(owner_thread_id_ == invalid_thread_id_)) {
      owner_thread_id_ = std::this_thread::get_id();
    } else {
      CHECK(likely(owner_thread_id_ == std::this_thread::get_id()));
    }
  }

  std::vector<T*> pool_;
  std::thread::id invalid_thread_id_;
  std::thread::id owner_thread_id_;
};

}  // namespace obj_pool
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SINGLE_THREAD_OBJ_POOL_H_
