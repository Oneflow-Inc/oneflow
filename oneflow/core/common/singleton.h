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
#ifndef ONEFLOW_CORE_COMMON_SINGLETON_H_
#define ONEFLOW_CORE_COMMON_SINGLETON_H_

#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/constant.h"

namespace oneflow {

template<typename T, typename Kind = void>
class Singleton final {
 public:
  static T* Get() { return *GetPPtr(); }
  static void SetAllocated(T* val) { *GetPPtr() = val; }
  template<typename... Args>
  static T* New(Args&&... args) {
    CHECK(Get() == nullptr);
    VLOG(3) << "NewGlobal " << typeid(T).name();
    T* ptr = new T(std::forward<Args>(args)...);
    *GetPPtr() = ptr;
    return ptr;
  }
  static void Delete() {
    if (Get() != nullptr) {
      VLOG(3) << "DeleteGlobal " << typeid(T).name();
      delete Get();
      *GetPPtr() = nullptr;
    }
  }

 private:
  static T** GetPPtr() {
    CheckKind();
    static T* ptr = nullptr;
    return &ptr;
  }
  static void CheckKind() {
    if (!std::is_same<Kind, void>::value) {
      CHECK(Singleton<T>::Get() == nullptr)
          << typeid(Singleton<T>).name() << " are disable for avoiding misuse";
    }
  }
};

template<typename T, typename... Kind>
Maybe<T*> SingletonMaybe() {
  CHECK_NOTNULL_OR_RETURN((Singleton<T, Kind...>::Get())) << " typeid: " << typeid(T).name();
  return Singleton<T, Kind...>::Get();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SINGLETON_H_
