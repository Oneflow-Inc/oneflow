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
#ifndef ONEFLOW_CORE_INTRUSIVE_OBJECT_POOL_H_
#define ONEFLOW_CORE_INTRUSIVE_OBJECT_POOL_H_

#include <vector>
#include "oneflow/core/intrusive/cpp_attribute.h"

namespace oneflow {
namespace intrusive {

enum ObjectPoolStrategey {
  kThreadUnsafeAndDisableDestruct,
};

template<typename T, ObjectPoolStrategey object_pool_strategy>
class ObjectPool;

template<typename T, ObjectPoolStrategey object_pool_strategy>
class EnableObjectPool {
 public:
  EnableObjectPool() = default;
  EnableObjectPool(const EnableObjectPool&) = default;
  EnableObjectPool(EnableObjectPool&&) = default;
  ~EnableObjectPool() = default;

  using object_pool_type = ObjectPool<T, object_pool_strategy>;
  object_pool_type* mut_object_pool() { return object_pool_; }
  void set_object_pool(object_pool_type* val) { object_pool_ = val; }

 private:
  object_pool_type* object_pool_;
};

template<typename T>
class ObjectPool<T, kThreadUnsafeAndDisableDestruct> {
 public:
  ObjectPool() { container_.reserve(kObjectPoolInitCap); }
  ObjectPool(const ObjectPool&) = delete;
  ObjectPool(ObjectPool&&) = delete;
  ~ObjectPool() {
    for (auto* elem : container_) { delete elem; }
  }

  template<typename... Args>
  intrusive::shared_ptr<T> make_shared(Args&&... args) {
    if (INTRUSIVE_PREDICT_FALSE(container_.empty())) {
      auto ptr = intrusive::make_shared<T>(std::forward<Args>(args)...);
      InitObjectPoolFields4Element(ptr.get());
      return ptr;
    } else {
      auto* ptr = container_.back();
      container_.pop_back();
      ptr->__Init__(std::forward<Args>(args)...);
      InitObjectPoolFields4Element(ptr);
      return intrusive::shared_ptr<T>(ptr);
    }
  }

  static void Put(void* raw_ptr) {
    T* ptr = reinterpret_cast<T*>(raw_ptr);
    ptr->mut_object_pool()->container_.push_back(ptr);
  }

 private:
  inline void InitObjectPoolFields4Element(T* ptr) {
    ptr->set_object_pool(this);
    ptr->mut_intrusive_ref()->set_deleter(&ObjectPool::Put);
  }

  static constexpr int kObjectPoolInitCap = 65536;
  std::vector<T*> container_;
};

}  // namespace intrusive
}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_OBJECT_POOL_H_
