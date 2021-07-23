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
#ifndef ONEFLOW_CORE_COMMON_GLOBAL_H_
#define ONEFLOW_CORE_COMMON_GLOBAL_H_

#include <mutex>
#include <map>
#include <memory>
#include <glog/logging.h>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/constant.h"

namespace oneflow {

template<typename T, typename Kind = void>
class Global final {
 public:
  static T* Get() { return *GetPPtr(); }
  static void SetAllocated(T* val) { *GetPPtr() = val; }
  template<typename... Args>
  static T* New(Args&&... args) {
    CHECK(Get() == nullptr);
    LOG(INFO) << "NewGlobal " << typeid(T).name();
    T* ptr = new T(std::forward<Args>(args)...);
    *GetPPtr() = ptr;
    return ptr;
  }
  static void Delete() {
    if (Get() != nullptr) {
      LOG(INFO) << "DeleteGlobal " << typeid(T).name();
      delete Get();
      *GetPPtr() = nullptr;
    }
  }
  // for each session
  static T* Get(int32_t session_id) {
    if (session_id == kInvalidSessionId) { return Get(); }
    return GetPPtr(session_id)->get();
  }
  static void SetAllocated(int32_t session_id, T* val) { GetPPtr(session_id)->reset(val); }
  template<typename... Args>
  static void SessionNew(int32_t session_id, Args&&... args) {
    CHECK(Get(session_id) == nullptr);
    LOG(INFO) << "session_id: " << session_id << ", NewGlobal " << typeid(T).name();
    GetPPtr(session_id)->reset(new T(std::forward<Args>(args)...));
  }
  static void SessionDelete(int32_t session_id) {
    if (Get(session_id) != nullptr) {
      LOG(INFO) << "session_id: " << session_id << ", DeleteGlobal " << typeid(T).name();
      GetPPtr(session_id)->reset();
    }
  }

 private:
  static T** GetPPtr() {
    CheckKind();
    static T* ptr = nullptr;
    return &ptr;
  }
  static std::unique_ptr<T>* GetPPtr(int32_t session_id) {
    CheckKind();
    static std::mutex mutex;
    static std::map<int32_t, std::unique_ptr<T>> session_id2ptr;
    std::unique_lock<std::mutex> lock(mutex);
    return &session_id2ptr[session_id];
  }
  static void CheckKind() {
    if (!std::is_same<Kind, void>::value) {
      CHECK(Global<T>::Get() == nullptr)
          << typeid(Global<T>).name() << " are disable for avoiding misuse";
    }
  }
};

template<typename T, typename... Kind>
Maybe<T*> GlobalMaybe() {
  CHECK_NOTNULL_OR_RETURN((Global<T, Kind...>::Get())) << " typeid: " << typeid(T).name();
  return Global<T, Kind...>::Get();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_GLOBAL_H_
