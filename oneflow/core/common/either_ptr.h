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

  using XPtr = std::shared_ptr<X>;
  using YPtr = std::shared_ptr<Y>;

  // WARNING: we should assume that the structure of shared_ptr<X> and shared_ptr<Y> is same,
  // and obviously at most time the assumption holds
  static_assert(sizeof(XPtr) == sizeof(YPtr), "unsupported shared_ptr implementation");

  EitherPtr() : type_(UnionType<X>::value), x_ptr_(nullptr) {}
  EitherPtr(const XPtr& ptr) : type_(UnionType<X>::value), x_ptr_(ptr) {}
  EitherPtr(const YPtr& ptr) : type_(UnionType<Y>::value) { new (&x_ptr_) YPtr(ptr); }

  EitherPtr(XPtr&& ptr) : type_(UnionType<X>::value), x_ptr_(std::move(ptr)) {}
  EitherPtr(YPtr&& ptr) : type_(UnionType<Y>::value) { new (&x_ptr_) YPtr(std::move(ptr)); }

  EitherPtr(const EitherPtr& either_ptr) : type_(either_ptr.type_), x_ptr_(either_ptr.x_ptr_) {}
  EitherPtr(EitherPtr&& either_ptr)
      : type_(either_ptr.type_), x_ptr_(std::move(either_ptr.x_ptr_)) {}

  // the destructor of X or Y will be called properly because it will be stored in the deleter of
  // shared_ptr while constructed
  ~EitherPtr() = default;

  EitherPtr& operator=(const EitherPtr& either_ptr) {
    x_ptr_ = either_ptr.x_ptr_;
    type_ = either_ptr.type_;
    return *this;
  }

  EitherPtr& operator=(EitherPtr&& either_ptr) {
    x_ptr_ = std::move(either_ptr.x_ptr_);
    type_ = either_ptr.type_;
    return *this;
  }

  template<typename T>
  bool Has() const {
    return type_ == UnionType<T>::value;
  }

  template<typename T>
  const std::shared_ptr<T>& Get() const {
    return Get(tag<T>{});
  }

 private:
  template<typename T, typename Enable = void>
  struct UnionType;
  template<typename T>
  struct UnionType<T, typename std::enable_if<std::is_same<X, T>::value>::type> {
    static constexpr int8_t value = 0;
  };
  template<typename T>
  struct UnionType<T, typename std::enable_if<std::is_same<Y, T>::value>::type> {
    static constexpr int8_t value = 1;
  };

  template<typename>
  struct tag {};

  const XPtr& Get(tag<X>) const {
    CHECK(Has<X>());
    return x_ptr_;
  }

  const YPtr& Get(tag<Y>) const {
    CHECK(Has<Y>());
    const auto* __attribute__((__may_alias__)) ptr = reinterpret_cast<const YPtr*>(&x_ptr_);
    return *ptr;
  }

  int8_t type_;
  std::shared_ptr<X> x_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EITHER_PTR_H_
