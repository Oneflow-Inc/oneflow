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
#ifndef ONEFLOW_CORE_COMMON_EITHER_PTR_H_
#define ONEFLOW_CORE_COMMON_EITHER_PTR_H_

#include <glog/logging.h>
#include <memory>

namespace oneflow {

template<typename X, typename Y>
class EitherPtr final {
 public:
  static_assert(!std::is_same<X, Y>::value, "X should not be Y");
  EitherPtr() : type_(UnionType<Void>::value) {}
  EitherPtr(const std::shared_ptr<X>& ptr) { Set(ptr); }
  EitherPtr(const std::shared_ptr<Y>& ptr) { Set(ptr); }
  EitherPtr(const EitherPtr<X, Y>& either_ptr) { CopyFrom(either_ptr); }
  ~EitherPtr() { Reset(); }

  template<typename T>
  bool Has() const {
    return type_ == UnionType<T>::value;
  }
  template<typename T>
  const std::shared_ptr<T>& Get() const {
    CHECK(this->template Has<T>());
    return Cast<T>();
  }
  void Reset(const std::shared_ptr<X>& ptr) {
    Reset();
    Set(ptr);
  }
  void Reset(const std::shared_ptr<Y>& ptr) {
    Reset();
    Set(ptr);
  }

  void Reset() {
    if (type_ == UnionType<Void>::value) {
      union_.reset();
    } else if (type_ == UnionType<X>::value) {
      MutCast<X>()->reset();
    } else if (type_ == UnionType<Y>::value) {
      MutCast<Y>()->reset();
    } else {
      LOG(FATAL) << "UNIMPLEMENTED";
    }
  }

 private:
  struct Void {};
  template<typename T, typename Enable = void>
  struct UnionType;
  template<typename T>
  struct UnionType<T, typename std::enable_if<std::is_same<Void, T>::value>::type> {
    static const int8_t value = 0;
  };
  template<typename T>
  struct UnionType<T, typename std::enable_if<std::is_same<X, T>::value>::type> {
    static const int8_t value = 1;
  };
  template<typename T>
  struct UnionType<T, typename std::enable_if<std::is_same<Y, T>::value>::type> {
    static const int8_t value = 2;
  };
  void CopyFrom(const EitherPtr<X, Y>& either_ptr) {
    if (either_ptr.template Has<X>()) {
      Set(either_ptr.template Get<X>());
    } else if (either_ptr.template Has<Y>()) {
      Set(either_ptr.template Get<Y>());
    } else {
      // do nothin
    }
  }
  void Set(const std::shared_ptr<X>& ptr) {
    CHECK(union_.get() == nullptr);
    *MutCast<X>() = ptr;
    type_ = UnionType<X>::value;
  }
  void Set(const std::shared_ptr<Y>& ptr) {
    CHECK(union_.get() == nullptr);
    *MutCast<Y>() = ptr;
    type_ = UnionType<Y>::value;
  }
  template<typename T>
  std::shared_ptr<T>* MutCast() {
    std::shared_ptr<T>* __attribute__((__may_alias__)) ptr =
        reinterpret_cast<std::shared_ptr<T>*>(&union_);
    return ptr;
  }
  template<typename T>
  const std::shared_ptr<T>& Cast() const {
    const std::shared_ptr<T>* __attribute__((__may_alias__)) ptr =
        reinterpret_cast<const std::shared_ptr<T>*>(&union_);
    return *ptr;
  }

  std::shared_ptr<Void> union_;
  int8_t type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EITHER_PTR_H_
